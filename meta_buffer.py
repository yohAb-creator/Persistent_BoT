import json
import numpy as np
from typing import Dict, List, Optional, Any


try:
    from sentence_transformers import SentenceTransformer
    import faiss

    RAG_ENABLED = True
except ImportError:
    print("Warning: 'sentence-transformers' or 'faiss-cpu' not found.")
    print("VectorMetaBuffer will operate in a fallback mode with no retrieval capabilities.")
    SentenceTransformer = None
    faiss = None
    RAG_ENABLED = False


class MetaBuffer:
    """
    Manages meta-reasoning templates using a vector database for efficient,
    optimal retrieval. Implements a Retrieval-Augmented Generation (RAG) system.
    Initial prototype retrieved every template.

    """

    def __init__(self, filepath: str = "meta_buffer_v4.json", embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the buffer, loads templates, and builds the vector index.
        """
        self.filepath = filepath
        self.templates: Dict[str, str] = {}
        self.vectors: Optional[np.ndarray] = None
        self.index: Optional[Any] = None  # faiss.Index
        self.embedding_model: Optional[Any] = None  # SentenceTransformer
        self.template_keys: List[str] = []

        if RAG_ENABLED:
            print(f"Initializing embedding model '{embedding_model_name}'...")
            self.embedding_model = SentenceTransformer(embedding_model_name)

        self._load()

    def _load(self):
        """Loads templates from the JSON file and builds the initial FAISS index."""
        try:
            with open(self.filepath, 'r') as f:
                self.templates = json.load(f)
            print(f"Loaded {len(self.templates)} templates from {self.filepath}.")
            if self.templates:
                self._rebuild_index()
        except (FileNotFoundError, json.JSONDecodeError):
            self.templates = {}
            print("No existing meta-buffer found. Initializing a new one.")

    def _save(self):
        """Saves the current set of templates to the JSON file."""
        with open(self.filepath, 'w') as f:
            json.dump(self.templates, f, indent=4)

    def _rebuild_index(self):
        """Encodes all templates and rebuilds the FAISS index for fast searching."""
        if not RAG_ENABLED or not self.templates or not self.embedding_model:
            return

        self.template_keys = list(self.templates.keys())
        # Use the content of the templates for embedding, not the keys
        template_contents = [self.templates[key] for key in self.template_keys]

        print("Embedding templates and rebuilding vector index...")
        self.vectors = self.embedding_model.encode(template_contents)
        embedding_dim = self.vectors.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(self.vectors.astype('float32'))
        print(f"FAISS index rebuilt successfully with {self.index.ntotal} vectors.")

    def add_templates(self, new_templates: Dict[str, str]):
        """
        Adds new templates to the buffer, saves to disk, and updates the live index.
        """
        num_added = sum(1 for key in new_templates if key not in self.templates)
        if num_added > 0:
            self.templates.update(new_templates)
            self._save()
            self._rebuild_index()  # Re-encode all templates to keep the index fresh
            print(f"Added {num_added} new template(s) and updated vector index.")

    def search(self, query_text: str, top_k: int = 3) -> Dict[str, str]:
        """
        Finds the most semantically similar templates to a given query text.
        """
        if not RAG_ENABLED or self.index is None or self.embedding_model is None or self.index.ntotal == 0:
            print("Warning: RAG system not available or index is empty. Returning no templates.")
            return {}

        if top_k <= 0:
            return {}

        # Encode the incoming query
        query_vector = self.embedding_model.encode([query_text])

        # Ensure k is not greater than the number of items in the index
        k = min(top_k, self.index.ntotal)

        # Search the FAISS index
        distances, indices = self.index.search(query_vector.astype('float32'), k)

        # Retrieve the original templates using the indices
        retrieved_keys = [self.template_keys[i] for i in indices[0]]
        retrieved_templates = {key: self.templates[key] for key in retrieved_keys}

        print(f"Retrieved {len(retrieved_templates)} templates for the query.")
        return retrieved_templates

    def get_all_templates(self) -> Dict[str, str]:
        """Returns all currently loaded templates."""
        return self.templates
