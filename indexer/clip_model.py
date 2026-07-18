# Colab-pasteable: yes
"""
CLIP Model Module.
Handles loading the pretrained CLIP model, generating visual/text embeddings, 
and performing zero-shot classifications. Pre-loads weights on import.
"""
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

try:
    from indexer.config import CLIP_MODEL_NAME
except ImportError:
    pass

_clip_model = None
_clip_processor = None

def load_clip():
    """Initializes and returns shared global instances of the CLIP Model and Processor."""
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        model_name = globals().get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = CLIPModel.from_pretrained(model_name).to(device)
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
    return _clip_model, _clip_processor

def embed_image(image: Image.Image) -> torch.Tensor:
    """Generates a normalized visual semantic embedding vector."""
    model, processor = load_clip()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(text=["image"], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.image_embeds
        features_norm = features / features.norm(dim=-1, keepdim=True)
    return features_norm.squeeze(0).cpu()

def embed_text(text: str) -> torch.Tensor:
    """Generates a normalized text semantic embedding vector using joint-model safe mapping."""
    model, processor = load_clip()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_img = Image.new("RGB", (1, 1), color=(0, 0, 0))
    inputs = processor(text=[text], images=dummy_img, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.text_embeds
        features_norm = features / features.norm(dim=-1, keepdim=True)
    return features_norm.squeeze(0).cpu()

def zero_shot_classify(image: Image.Image, candidate_labels: list) -> list:
    """Performs zero-shot image classification across arbitrary text prompts."""
    model, processor = load_clip()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = [f"a photo of a {label}" for label in candidate_labels]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        image_features_norm = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_features_norm = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        scores = (image_features_norm @ text_features_norm.T).squeeze(0).cpu().tolist()
    return sorted(zip(candidate_labels, scores), key=lambda x: x[1], reverse=True)

try:
    load_clip()
except Exception as e:
    print(f"[CLIP] Warmup skipped: {e}")