"""
Wrapper class for FAISS indexing and JSON payload storage.
Includes automatic path normalization to seamlessly map absolute Colab/Drive paths 
to local relative paths during search.
"""
import os
import json
import numpy as np
import faiss

class FashionVectorStore:
    """Handles IO disk read/write mappings for FAISS indexes and payload dictionaries."""
    def __init__(self, vector_dim=512, output_dir="./vector_store/output"):
        self.vector_dim = vector_dim
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.global_index_path = os.path.join(self.output_dir, "global_index.faiss")
        self.local_index_path = os.path.join(self.output_dir, "local_index.faiss")
        self.global_meta_path = os.path.join(self.output_dir, "global_metadata.json")
        self.local_meta_path = os.path.join(self.output_dir, "local_metadata.json")
        
        self.global_index = faiss.IndexFlatIP(self.vector_dim)
        self.local_index = faiss.IndexFlatIP(self.vector_dim)
        
        self.global_metadata = []
        self.local_metadata = []
        self.load_index()

    def _normalize_path(self, original_path: str) -> str:
        """Dynamically converts absolute Colab/Drive paths to standard local paths."""
        filename = os.path.basename(original_path)
        return os.path.join("data/images", filename).replace("\\", "/")

    def load_index(self):
        """Loads index weights and JSON payloads from disk, normalizing paths on the fly."""
        if os.path.exists(self.global_index_path):
            self.global_index = faiss.read_index(self.global_index_path)
        if os.path.exists(self.local_index_path):
            self.local_index = faiss.read_index(self.local_index_path)
            
        if os.path.exists(self.global_meta_path):
            with open(self.global_meta_path, "r") as f:
                raw_meta = json.load(f)
                for item in raw_meta:
                    item["image_path"] = self._normalize_path(item["image_path"])
                self.global_metadata = raw_meta
                
        if os.path.exists(self.local_meta_path):
            with open(self.local_meta_path, "r") as f:
                raw_meta = json.load(f)
                for item in raw_meta:
                    item["image_path"] = self._normalize_path(item["image_path"])
                self.local_metadata = raw_meta

    def add_global_vectors(self, vectors: np.ndarray, metadata: list):
        if len(vectors) == 0: return
        self.global_index.add(np.array(vectors).astype('float32'))
        self.global_metadata.extend(metadata)

    def add_local_vectors(self, vectors: np.ndarray, metadata: list):
        if len(vectors) == 0: return
        self.local_index.add(np.array(vectors).astype('float32'))
        self.local_metadata.extend(metadata)

    def save_index(self):
        """Saves memory-held vector changes directly back to index storage files."""
        faiss.write_index(self.global_index, self.global_index_path)
        faiss.write_index(self.local_index, self.local_index_path)
        
        with open(self.global_meta_path, "w") as f:
            json.dump(self.global_metadata, f, indent=2)
        with open(self.local_meta_path, "w") as f:
            json.dump(self.local_metadata, f, indent=2)
        print(f"[Store] Saved indices and payload configurations to '{self.output_dir}'.")