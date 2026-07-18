# Colab-pasteable: yes
"""
Color Extraction Module.
Uses statistical random pixel subsampling (max=1000) to keep K-Means execution time under 2ms.
"""
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import torch
import webcolors

try:
    from indexer.clip_model import load_clip
except ImportError:
    pass

BASIC_COLORS = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "brown", "black", "white", "gray", "beige", "navy", "olive"
]

CSS3_RGB_MAP = {}
try:
    css3_names = webcolors.names(spec='css3')
    for name in css3_names:
        rgb = webcolors.name_to_rgb(name, spec='css3')
        CSS3_RGB_MAP[name] = (rgb.red, rgb.green, rgb.blue)
except Exception as e:
    print(f"[Warning] Loading webcolors mapping failed ({str(e)}). Using static fallback.")
    CSS3_RGB_MAP = {
        "white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128),
        "red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0),
        "navy": (0, 0, 128), "olive": (128, 128, 0), "yellow": (255, 255, 0),
        "orange": (255, 127, 0), "purple": (128, 0, 128), "pink": (255, 192, 203)
    }


def get_precise_color(crop_image: Image.Image, mask_segment: np.ndarray) -> str:
    """Extracts dominant precise CSS3 color name using K-Means (k=3)."""
    crop_np = np.array(crop_image.convert("RGB"))
    h, w, c = crop_np.shape

    mask_uint8 = (mask_segment.astype(np.uint8)) * 255
    mask_resized = Image.fromarray(mask_uint8).resize((w, h), resample=Image.Resampling.NEAREST)
    mask_np = np.array(mask_resized) > 0

    masked_pixels = crop_np[mask_np]
    total_masked = len(masked_pixels)

    if total_masked == 0:
        masked_pixels = crop_np.reshape(-1, 3)
        total_masked = len(masked_pixels)

    brightness = masked_pixels.mean(axis=1)
    valid_pixels = masked_pixels[(brightness > 20) & (brightness < 235)]

    if len(valid_pixels) >= 0.3 * total_masked and len(valid_pixels) > 0:
        filtered_pixels = valid_pixels
    else:
        filtered_pixels = masked_pixels

    num_samples = len(filtered_pixels)

    if num_samples > 1000:
        indices = np.random.choice(num_samples, 1000, replace=False)
        filtered_pixels = filtered_pixels[indices]
        num_samples = 1000

    print(f"  [Color Filtering] Subsampled {num_samples} of {total_masked} masked pixels.")

    n_clusters = min(3, num_samples)
    if n_clusters == 0:
        return "white"

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(filtered_pixels)

    labels = kmeans.labels_
    counts = np.bincount(labels)
    dominant_centroid = kmeans.cluster_centers_[np.argmax(counts)]
    dominant_rgb = [int(val) for val in dominant_centroid]

    closest_name = "white"
    min_dist = float('inf')
    for name, rgb_val in CSS3_RGB_MAP.items():
        dist = np.linalg.norm(np.array(dominant_rgb) - np.array(rgb_val))
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name


def get_broad_color(crop_image: Image.Image) -> str:
    """Classifies a broad color category using zero-shot CLIP matching on the cropped region."""
    clip_model_func = globals().get("load_clip", None)
    if clip_model_func is None:
        try:
            from indexer.clip_model import load_clip as clip_model_func
        except ImportError:
            pass

    model, processor = clip_model_func()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    color_prompts = [f"a photo of a {color} garment" for color in BASIC_COLORS]
    inputs = processor(text=color_prompts, images=crop_image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1).squeeze(0).tolist()

    best_idx = np.argmax(probs)
    return BASIC_COLORS[best_idx]