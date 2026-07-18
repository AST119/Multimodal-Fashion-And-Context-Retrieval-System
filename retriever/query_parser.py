"""
Query Parser Module.
Parses natural language queries into global scene, local garments, colors, and design attributes.
Implements a high-precision Clause-Split Parser to prevent color cross-contamination,
with an optional LangChain Structured LLM parsing mode.
"""
import os
import re
from retriever.color_bucket_map import BASIC_BUCKETS
from indexer.vocabulary import VOCABULARY, CONTEXT_AXES

try:
    from pydantic import BaseModel, Field
    from typing import List, Optional
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

FASHION_ATTRIBUTES_MAP = {}
try:
    from indexer.config import FASHION_ATTRIBUTES_MAP
except ImportError:
    pass

# Dynamically compile the flat attributes search space from FASHION_ATTRIBUTES_MAP
ALL_ATTRIBUTES = []
for cat, sub_maps in FASHION_ATTRIBUTES_MAP.items():
    for attr_type, values in sub_maps.items():
        for val in values:
            clean = val.replace(" jacket", "").replace(" coat", "").replace(" dress", "").replace(" pants", "").replace(" skirt", "").replace(" t-shirt", "").replace(" shirt", "").replace(" pocket", "")
            ALL_ATTRIBUTES.append(clean.strip())
ALL_ATTRIBUTES = list(set(ALL_ATTRIBUTES))

SYNONYM_MAP = {
    "raincoat": "coat",
    "trenchcoat": "coat",
    "trench": "coat",
    "blazer": "jacket",
    "t-shirt": "top, t-shirt, sweatshirt",
    "tshirt": "top, t-shirt, sweatshirt",
    "sweatshirt": "top, t-shirt, sweatshirt",
    "hoodie": "top, t-shirt, sweatshirt",
    "jeans": "pants, trousers, jeans",
    "trousers": "pants, trousers, jeans",
    "gown": "evening gown, formal dress, cocktail dress",
    "clutch": "wallet, clutch",
    "handbag": "bag, handbag, tote",
    "tote": "bag, handbag, tote"
}

# Regex patterns to strip semantic noise from scene context
FILLER_PATTERNS = [
    re.compile(r"\b(someone|a person|model|man|woman|girl|boy|people)\b", re.IGNORECASE),
    re.compile(r"\b(wearing|in|dressed in|attire for|outfit for|with)\s+a?\b", re.IGNORECASE),
    re.compile(r"\b(sitting on|inside|at|on|for a)\s+a?\b", re.IGNORECASE),
]

GARMENT_PATTERNS = []
for g in VOCABULARY:
    for part in g.split(','):
        part_clean = part.strip()
        if part_clean:
            GARMENT_PATTERNS.append((re.compile(rf"\b{re.escape(part_clean)}s?\b", re.IGNORECASE), g))

ATTRIBUTE_PATTERNS = [
    (re.compile(rf"\b{re.escape(attr)}\b", re.IGNORECASE), attr)
    for attr in ALL_ATTRIBUTES if attr
]

def lemmatize_action_query(query_str: str) -> str:
    """Normalizes inflected active or past action verbs to their base form."""
    joined = re.sub(r'\b(sitting|sat|sits)\b', 'sit', query_str)
    joined = re.sub(r'\b(standing|stood|stands)\b', 'stand', joined)
    joined = re.sub(r'\b(walking|walked|walks)\b', 'walk', joined)
    joined = re.sub(r'\b(running|ran|runs)\b', 'run', joined)
    joined = re.sub(r'\b(posing|posed|poses)\b', 'pose', joined)
    return joined

# Pydantic Schemas for Structured LLM Parsing
if HAS_LANGCHAIN:
    class GarmentItem(BaseModel):
        garment: str = Field(description="Canonical garment category matching our vocabulary.")
        color: Optional[str] = Field(None, description="Color associated with this garment.")
        attributes: List[str] = Field(default_factory=list, description="Fashion attributes like 'puffer', 'maxi', 'biker'.")

    class StructuredQuery(BaseModel):
        scene_phrase: str = Field(description="Background setting, location, weather, or overall vibe (e.g. 'park bench', 'office').")
        setting_places: Optional[str] = Field(None, description=f"Closest match from: {CONTEXT_AXES['places']}")
        setting_seasons: Optional[str] = Field(None, description=f"Closest match from: {CONTEXT_AXES['seasons']}")
        setting_vibes: Optional[str] = Field(None, description=f"Closest match from: {CONTEXT_AXES['vibes']}")
        setting_actions: Optional[str] = Field(None, description=f"Closest match from: {CONTEXT_AXES['actions']}")
        items: List[GarmentItem] = Field(default_factory=list, description="List of detected garment items.")

def parse_query(query_str: str) -> dict:
    """Parses queries using either LangChain Structured Output or local Clause-Split fallback."""
    query_str_normalized = lemmatize_action_query(query_str)
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if HAS_LANGCHAIN and openai_key:
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
            structured_llm = llm.with_structured_output(StructuredQuery)
            
            system_prompt = (
                "You are an expert fashion query parser. Extract the visual scene context (location, background, vibe, action) "
                "and the specific list of garment items, including their color and visual attributes.\n"
                f"Valid Garments: {VOCABULARY}\n"
                f"Valid Colors: {BASIC_BUCKETS}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Parse this query: {query}")
            ])
            
            chain = prompt | structured_llm
            result = chain.invoke({"query": query_str_normalized})
            
            pydantic_dict = result.model_dump()
            pydantic_dict["setting"] = {
                "place": pydantic_dict.pop("setting_places"),
                "season": pydantic_dict.pop("setting_seasons"),
                "vibe": pydantic_dict.pop("setting_vibes"),
                "action": pydantic_dict.pop("setting_actions")
            }
            return pydantic_dict
        except Exception as e:
            pass
    
    query_str_lower = query_str_normalized.lower().strip()
    for syn, canonical in SYNONYM_MAP.items():
        query_str_lower = re.sub(rf"\b{re.escape(syn)}\b", canonical, query_str_lower)

    # Split query into independent visual clauses to isolate local attributes
    clauses = re.split(r'\b(?:and|with|wearing|dressed in|paired with)\b', query_str_lower)
    detected_items = []
    scene_pieces = []
    
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
            
        clause_colors = [color for color in BASIC_BUCKETS if re.search(rf"\b{re.escape(color)}\b", clause)]
        
        clause_garments = []
        for pattern, canonical_name in GARMENT_PATTERNS:
            match = pattern.search(clause)
            if match:
                clause_garments.append((match.start(), match.end(), canonical_name))
        
        clause_garments = sorted(clause_garments, key=lambda x: (x[1] - x[0]), reverse=True)
        non_overlapping = []
        for item in clause_garments:
            start, end, name = item
            if not any(start < o_end and end > o_start for o_start, o_end, _ in non_overlapping):
                non_overlapping.append(item)
                
        clause_attributes = []
        for pattern, attr_name in ATTRIBUTE_PATTERNS:
            if pattern.search(clause):
                clause_attributes.append(attr_name)
        
        for start, end, name in non_overlapping:
            color_match = None
            if clause_colors:
                min_dist = float('inf')
                for color in clause_colors:
                    color_pos = clause.find(color)
                    dist = abs(start - color_pos)
                    if dist < min_dist:
                        min_dist = dist
                        color_match = color
            
            item_attributes = []
            for attr in clause_attributes:
                attr_pos = clause.find(attr)
                if abs(start - attr_pos) < 20:
                    item_attributes.append(attr)
                    
            detected_items.append({
                "garment": name,
                "color": color_match,
                "attributes": item_attributes
            })
            
        clean_clause = clause
        for color in clause_colors:
            clean_clause = re.sub(rf"\b{re.escape(color)}\b", "", clean_clause)
        for _, _, name in non_overlapping:
            for part in name.split(','):
                clean_clause = re.sub(rf"\b{re.escape(part.strip())}s?\b", "", clean_clause, flags=re.IGNORECASE)
        for attr in clause_attributes:
            clean_clause = re.sub(rf"\b{re.escape(attr)}\b", "", clean_clause, flags=re.IGNORECASE)
            
        for pattern in FILLER_PATTERNS:
            clean_clause = pattern.sub("", clean_clause)
            
        clean_clause = re.sub(r'\s+', ' ', clean_clause).strip()
        if clean_clause:
            scene_pieces.append(clean_clause)
            
    scene_phrase = " ".join(scene_pieces)
    scene_phrase = re.sub(r'^(on|in|at|with|a|an|the)\b', '', scene_phrase, flags=re.IGNORECASE).strip()
    
    bas_place = next((p for p in CONTEXT_AXES["places"] if any(w in scene_phrase for w in p.split(", "))), None)
    bas_season = next((s for s in CONTEXT_AXES["seasons"] if any(w in scene_phrase for w in s.split())), None)
    bas_vibe = next((v for v in CONTEXT_AXES["vibes"] if any(w in scene_phrase for w in v.split(", "))), None)
    bas_action = next((a for a in CONTEXT_AXES["actions"] if a in scene_phrase), None)

    return {
        "scene_phrase": scene_phrase if scene_phrase else "fashion clothing model",
        "setting": {
            "place": bas_place,
            "season": bas_season,
            "vibe": bas_vibe,
            "action": bas_action
        },
        "items": detected_items
    }