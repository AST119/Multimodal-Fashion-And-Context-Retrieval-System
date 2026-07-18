# Colab-pasteable: yes
"""
Segmentation Module.
Exclusively utilizes CLIPSeg (CIDAS/clipseg-rd64-refined) for lightweight, 
extremely fast, and ungated instance segmentation on GPU. Pre-loads weights on import.
"""
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

try:
    from indexer.config import CLIPSEG_MODEL_NAME, CLIPSEG_CONF_THRESHOLD
except ImportError:
    pass

_model = None
_processor = None

def load_clipseg():
    """Initializes and ports the CLIPSeg model directly to the GPU (CUDA)."""
    global _model, _processor
    if _model is not None:
        return
    
    model_name = globals().get("CLIPSEG_MODEL_NAME", "CIDAS/clipseg-rd64-refined")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(device)
    _processor = CLIPSegProcessor.from_pretrained(model_name)
    print(f"[Segmenter] Loaded public CLIPSeg model '{model_name}' successfully on {device}.")

def run_clipseg(image: Image.Image, prompt: str) -> list:
    """Executes promptable segmentation using CLIPSeg."""
    load_clipseg()
    conf_threshold = globals().get("CLIPSEG_CONF_THRESHOLD", 0.5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    inputs = _processor(text=[prompt], images=[image], padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    
    with torch.no_grad():
        outputs = _model(**inputs)

    orig_w, orig_h = image.size
    logits = outputs.logits
    
    while logits.ndim < 4:
        logits = logits.unsqueeze(0)
    while logits.ndim > 4:
        logits = logits.squeeze(0)

    resized_logits = F.interpolate(
        logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    ).squeeze()

    mask_probs = torch.sigmoid(resized_logits).cpu().numpy()
    mask = mask_probs > 0.4
    
    y_indices, x_indices = np.where(mask)
    if len(x_indices) > 0:
        box = [
            float(np.min(x_indices)), 
            float(np.min(y_indices)), 
            float(np.max(x_indices)), 
            float(np.max(y_indices))
        ]
        score = float(np.mean(mask_probs[mask]))
        if score >= conf_threshold:
            results.append({
                "bbox": box,
                "confidence": score,
                "mask": mask
            })
    return results

try:
    load_clipseg()
except Exception as e:
    print(f"[Segmenter] Warmup skipped: {e}")