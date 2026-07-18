# Colab-pasteable: yes
"""
===========================================================================
                               COLAB USAGE
===========================================================================
To run this indexing pipeline at scale using a Colab GPU:

1. Create a fresh notebook in Google Colab and set the runtime to GPU.
2. Create sequential code cells and paste these files in order:
   Cell 1: indexer/vocabulary.py
   Cell 2: indexer/config.py
   Cell 3: indexer/clip_model.py
   Cell 4: indexer/clipseg_model.py
   Cell 5: indexer/color_extraction.py
   Cell 6: indexer/build_index.py
3. Install required packages in Colab:
   !pip install transformers torch numpy pillow scikit-learn webcolors faiss-cpu
4. Run cells top-to-bottom.
===========================================================================
"""
import os
import json
from PIL import Image
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

# Enable PyTorch hardware-level CUDA auto-tuners
torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)

try:
    import indexer.config as config
    from indexer.config import FASHION_ATTRIBUTES_MAP
    from indexer.vocabulary import VOCABULARY, CONTEXT_AXES
    from indexer.clip_model import load_clip, embed_image, zero_shot_classify
    from indexer.clipseg_model import run_clipseg
    from indexer.color_extraction import get_precise_color, get_broad_color
except ImportError:
    class ConfigFallback:
        IMAGE_DIR = globals().get("IMAGE_DIR", "./data/images")
        OUTPUT_DIR = globals().get("OUTPUT_DIR", "./vector_store/output")
        CROPS_DIR = globals().get("CROPS_DIR", "./crops")
        CLIP_PREFILTER_THRESHOLD = globals().get("CLIP_PREFILTER_THRESHOLD", 0.22)
        CLIP_PREFILTER_TOP_K = globals().get("CLIP_PREFILTER_TOP_K", 5)
    config = ConfigFallback()

attribute_map_global = globals().get("FASHION_ATTRIBUTES_MAP", {})
if not attribute_map_global:
    try:
        from indexer.config import FASHION_ATTRIBUTES_MAP as attribute_map_global
    except ImportError:
        attribute_map_global = {}


def async_cpu_tasks(crop_img, mask, cat, bbox, local_vec, broad_color, detected_attributes, confidence, img_name, img_path, crops_dir, precise_color_func):
    """Runs CPU-bound color extraction and saving asynchronously in background."""
    category_clean = cat.replace(", ", "_").replace(" ", "_")
    crop_name = f"{os.path.splitext(img_name)[0]}_{category_clean}.jpg"
    crop_path = os.path.join(crops_dir, crop_name)
    crop_img.save(crop_path)

    precise_color = precise_color_func(crop_img, mask)

    metadata = {
        "parent_image": img_name,
        "image_path": img_path,
        "category": cat,
        "bbox": bbox,
        "precise_color": precise_color,
        "broad_color": broad_color,
        "attributes": detected_attributes,
        "confidence": confidence
    }
    return local_vec, metadata


def run_indexing_pipeline():
    img_dir = getattr(config, "IMAGE_DIR", "./data/images")
    out_dir = getattr(config, "OUTPUT_DIR", "./vector_store/output")
    crops_dir = getattr(config, "CROPS_DIR", "./crops")
    prefilter_thresh = getattr(config, "CLIP_PREFILTER_THRESHOLD", 0.22)
    prefilter_top_k = getattr(config, "CLIP_PREFILTER_TOP_K", 5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_list = globals().get("VOCABULARY", None)
    if vocab_list is None:
        try:
            from indexer.vocabulary import VOCABULARY as vocab_list
        except ImportError:
            vocab_list = []

    axes_map = globals().get("CONTEXT_AXES", None)
    if axes_map is None:
        try:
            from indexer.vocabulary import CONTEXT_AXES as axes_map
        except ImportError:
            axes_map = {}

    print(f"\n[Indexer] Starting indexing pipeline on directory: '{img_dir}'")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    total_images = len(image_files)
    print(f"[Indexer] Found {total_images} target images for processing.")

    try:
        from vector_store.store import FashionVectorStore
        store = FashionVectorStore(vector_dim=512, output_dir=out_dir)
    except Exception as e:
        print(f"[Warning] Could not initialize FashionVectorStore ({str(e)}). Running local fallback saving.")
        store = None

    global_embeddings = []
    global_metadata = []
    local_embeddings = []
    local_metadata = []

    executor = ThreadPoolExecutor(max_workers=4)
    background_tasks = []

    z_shot_func = globals().get("zero_shot_classify", None)
    embed_func = globals().get("embed_image", None)
    clipseg_func = globals().get("run_clipseg", None)
    precise_color_func = globals().get("get_precise_color", None)
    broad_color_func = globals().get("get_broad_color", None)

    if z_shot_func is None:
        from indexer.clip_model import zero_shot_classify as z_shot_func, embed_image as embed_func
        from indexer.clipseg_model import run_clipseg as clipseg_func
        from indexer.color_extraction import get_precise_color as precise_color_func, get_broad_color as broad_color_func
        
    clip_model, clip_processor = load_clip()

    with torch.inference_mode():
        for idx, img_name in enumerate(image_files):
            img_path = os.path.join(img_dir, img_name)
            print(f"\nProcessing image {idx + 1}/{total_images}: '{img_name}'")
            
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  [Error] Failed to load '{img_name}': {str(e)}")
                continue

            axes_labels = {}
            for axis_name, candidates in axes_map.items():
                classification = z_shot_func(image, candidates)
                axes_labels[axis_name] = classification[0][0]
            
            print(f"  [Global Context] Place: '{axes_labels.get('places')}' | Season: '{axes_labels.get('seasons')}' | Vibe: '{axes_labels.get('vibes')}' | Action: '{axes_labels.get('actions')}'")

            # CLIP zero-shot pre-filter
            classifications = z_shot_func(image, vocab_list)
            surviving_categories = []
            skipped_categories = []
            for rank, (cat, score) in enumerate(classifications):
                if score >= prefilter_thresh or rank < prefilter_top_k:
                    surviving_categories.append((cat, score))
                else:
                    skipped_categories.append(cat)
            
            print(f"  [CLIP Pre-filter] Kept {len(surviving_categories)} classes. Skipped {len(skipped_categories)} classes.")

            global_vec = embed_func(image).numpy()
            global_embeddings.append(global_vec)
            global_metadata.append({
                "image_path": img_path, 
                "image_name": img_name,
                "place": axes_labels.get("places"),
                "season": axes_labels.get("seasons"),
                "vibe": axes_labels.get("vibes"),
                "action": axes_labels.get("actions")
            })

            for cat, score in surviving_categories:
                detections = clipseg_func(image, cat)
                if not detections:
                    continue
                
                print(f"  [Segmenter] Detected {len(detections)} instance(s) of '{cat}'.")
                for d_idx, det in enumerate(detections):
                    bbox = det["bbox"]
                    confidence = det["confidence"]
                    mask = det["mask"]

                    crop_img = image.crop(bbox)

                    crop_inputs = clip_processor(text=["garment"], images=crop_img, return_tensors="pt", padding=True)
                    crop_inputs = {k: v.to(device) for k, v in crop_inputs.items()}
                    crop_out = clip_model(**crop_inputs)
                    crop_features = crop_out.image_embeds
                    crop_features_norm = crop_features / crop_features.norm(dim=-1, keepdim=True)
                    local_vec = crop_features_norm.squeeze(0).cpu().numpy()

                    broad_color = broad_color_func(crop_img)

                    # Hierarchical Attribute Tagging (Confidence Gated)
                    detected_attributes = []
                    attribute_map = globals().get("FASHION_ATTRIBUTES_MAP", attribute_map_global)
                    
                    if cat in attribute_map:
                        flat_candidates = []
                        attr_types_boundaries = []
                        start_idx = 0
                        
                        for attr_type, candidate_values in attribute_map[cat].items():
                            flat_candidates.extend(candidate_values)
                            end_idx = start_idx + len(candidate_values)
                            attr_types_boundaries.append((attr_type, start_idx, end_idx, candidate_values))
                            start_idx = end_idx
                        
                        if flat_candidates:
                            attr_prompts = [f"a photo of a {val}" for val in flat_candidates]
                            attr_inputs = clip_processor(text=attr_prompts, images=crop_img, return_tensors="pt", padding=True)
                            attr_inputs = {k: v.to(device) for k, v in attr_inputs.items()}
                            
                            attr_outputs = clip_model(**attr_inputs)
                            probs = attr_outputs.logits_per_image.softmax(dim=-1).squeeze(0).cpu().numpy()
                            
                            for attr_type, start, end, candidate_values in attr_types_boundaries:
                                subset_probs = probs[start:end]
                                best_idx = np.argmax(subset_probs)
                                best_prob = subset_probs[best_idx]
                                
                                if best_prob >= 0.35:
                                    best_attr = candidate_values[best_idx]
                                    clean_attr = best_attr.replace(" jacket", "").replace(" coat", "").replace(" dress", "").replace(" pants", "").replace(" skirt", "").replace(" t-shirt", "").replace(" shirt", "").replace(" pocket", "")
                                    if clean_attr != "plain":
                                        detected_attributes.append(clean_attr)

                    task = executor.submit(
                        async_cpu_tasks,
                        crop_img, mask, cat, bbox, local_vec, broad_color, 
                        detected_attributes, confidence, img_name, img_path, 
                        crops_dir, precise_color_func
                    )
                    background_tasks.append(task)

    print("\n[Indexer] Finalizing background color extraction and disk operations...")
    for task in background_tasks:
        local_vec, meta_record = task.result()
        local_embeddings.append(local_vec)
        local_metadata.append(meta_record)

    executor.shutdown()

    if store is not None:
        store.add_global_vectors(np.array(global_embeddings), global_metadata)
        store.add_local_vectors(np.array(local_embeddings), local_metadata)
        store.save_index()
    else:
        import faiss
        global_index = faiss.IndexFlatIP(512)
        if len(global_embeddings) > 0:
            global_index.add(np.array(global_embeddings).astype('float32'))
        faiss.write_index(global_index, os.path.join(out_dir, "global_index.faiss"))
        with open(os.path.join(out_dir, "global_metadata.json"), "w") as f:
            json.dump(global_metadata, f, indent=2)

        local_index = faiss.IndexFlatIP(512)
        if len(local_embeddings) > 0:
            local_index.add(np.array(local_embeddings).astype('float32'))
        faiss.write_index(local_index, os.path.join(out_dir, "local_index.faiss"))
        with open(os.path.join(out_dir, "local_metadata.json"), "w") as f:
            json.dump(local_metadata, f, indent=2)

    print(f"\n[Indexer] Indexing completed. Saved outputs to: '{out_dir}'")
    for f in os.listdir(out_dir):
        f_path = os.path.join(out_dir, f)
        print(f"  - {f} ({os.path.getsize(f_path) / 1024:.2f} KB)")

if __name__ == "__main__":
    run_indexing_pipeline()