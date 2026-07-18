"""
Search Module.
Performs dynamic, multiplicative multi-vector search using Global (Scene) and Local (Garment) indices.
"""
import numpy as np
import faiss
import os

from indexer.clip_model import embed_text
from retriever.query_parser import parse_query
from retriever.color_bucket_map import COLOR_BUCKET_MAP
from vector_store.store import FashionVectorStore

def search_fashion_database(query_str: str, k=10) -> list:
    """Performs visual-semantic search using dynamic multiplicative weights and metadata setting matching."""
    store = FashionVectorStore(vector_dim=512, output_dir="./vector_store/output")
    
    if len(store.global_metadata) == 0 or store.global_index.ntotal == 0:
        print("[Search] Index files are empty. Running build_index.py is required first.")
        return []

    parsed = parse_query(query_str)
    scene_phrase = parsed["scene_phrase"]
    query_items = parsed["items"]
    query_setting = parsed["setting"]

    print(f"\n[Search] Executing search for: '{query_str}'")
    print(f"  - Context Component: '{scene_phrase}'")
    print(f"  - Isolated Garments: {query_items}")
    print(f"  - Setting Criteria: {query_setting}")

    num_items = len(query_items)
    if num_items == 0:
        weight_global = 1.0
        weight_local = 0.0
    else:
        weight_global = 1.0 / (1.0 + num_items)
        weight_local = 1.0 - weight_global

    # 1. Whole-Image Context Similarity Search
    scene_vector = embed_text(scene_phrase).numpy().reshape(1, -1).astype('float32')
    num_global = store.global_index.ntotal
    g_distances, g_indices = store.global_index.search(scene_vector, num_global)
    
    image_scores = {}
    for dist, idx in zip(g_distances[0], g_indices[0]):
        if idx == -1: continue
        meta = store.global_metadata[idx]
        img_name = meta["image_name"]
        img_path = meta["image_path"]
        
        # Calculate context-axes intersection metadata score boost
        axis_score_boost = 0.0
        for axis_key in ["place", "season", "vibe", "action"]:
            target_val = query_setting.get(axis_key)
            stored_val = meta.get(axis_key)
            if target_val and stored_val and target_val == stored_val:
                axis_score_boost += 0.15  # 15% score boost per matching context-axis
                
        image_scores[img_name] = {
            "image_path": img_path,
            "scene_score": float(dist) + axis_score_boost,
            "garment_scores": [],
            "matches": []
        }

    # 2. Segment-Level Content Searches with Dual-Layer Color & Attribute Matching
    if store.local_index.ntotal > 0 and num_items > 0:
        for item in query_items:
            garment_type = item["garment"]
            target_color = item["color"]
            target_attrs = item.get("attributes", [])
            
            # Build query string
            color_str = f"{target_color} " if target_color else ""
            attr_str = f"{' '.join(target_attrs)} " if target_attrs else ""
            item_query = f"a photo of a {color_str}{attr_str}{garment_type}".replace("  ", " ")
            
            item_vector = embed_text(item_query).numpy().reshape(1, -1).astype('float32')
            
            num_local = store.local_index.ntotal
            l_distances, l_indices = store.local_index.search(item_vector, num_local)
            
            for dist, idx in zip(l_distances[0], l_indices[0]):
                if idx == -1: continue
                meta = store.local_metadata[idx]
                parent_img = meta["parent_image"]
                
                if parent_img not in image_scores:
                    continue
                if meta["category"] != garment_type:
                    continue
                
                attribute_score_boost = 0.0
                stored_attrs = meta.get("attributes", [])
                if target_attrs:
                    matches = set(target_attrs).intersection(set(stored_attrs))
                    attribute_score_boost = len(matches) * 0.15
                
                # Dual-layered color validations
                color_matched = False
                if target_color:
                    precise = meta["precise_color"]
                    broad = meta["broad_color"]
                    
                    if target_color == broad or target_color == precise:
                        color_matched = True
                    elif COLOR_BUCKET_MAP.get(precise, "") == target_color:
                        color_matched = True
                else:
                    color_matched = True
                    
                if color_matched:
                    final_match_score = float(dist) + attribute_score_boost
                    image_scores[parent_img]["garment_scores"].append(final_match_score)
                    image_scores[parent_img]["matches"].append({
                        "category": garment_type,
                        "precise_color": meta["precise_color"],
                        "broad_color": meta["broad_color"],
                        "attributes": stored_attrs,
                        "bbox": meta["bbox"]
                    })
                    break

    # 3. Advanced Multiplicative Score Fusion
    results = []
    for name, data in image_scores.items():
        g_score = data["scene_score"]
        l_scores = data["garment_scores"]
        actual_matches = len(l_scores)
        
        if num_items == 0:
            fused_score = g_score
        else:
            # COMPOSITIONAL SAFEGUARD (Missing Item Penalty)
            avg_local_score = sum(l_scores) / num_items if actual_matches > 0 else 0.0
            
            # ADVANCED MULTIPLICATIVE FUSION (Coordinate Mapping):
            # Temporarily shift both scores to the [0, 2] range, apply the weights exponentially,
            # multiply them, and then shift back. This collapses the final rank if either setting or garment fails.
            g_shifted = max(0.0, g_score + 1.0)
            l_shifted = max(0.0, avg_local_score + 1.0)
            
            fused_shifted = (g_shifted ** weight_global) * (l_shifted ** weight_local)
            fused_score = fused_shifted - 1.0
            
        results.append({
            "image_name": name,
            "image_path": data["image_path"],
            "score": round(fused_score, 4),
            "matched_garments": data["matches"]
        })
        
    return sorted(results, key=lambda x: x["score"], reverse=True)[:k]