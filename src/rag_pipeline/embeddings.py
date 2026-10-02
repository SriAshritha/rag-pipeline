from .chunking import chunk_files
from .documents_load import load_documents
from sentence_transformers import SentenceTransformer
import numpy as np

def generate_embeddings(chunks: str) -> np.ndarray:
    load_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = load_model.encode(texts,show_progress_bar = True)
    print(f"Total number of embeddings : {len(embeddings)} and {len(embeddings[0])} is the dimension of each embedding")

    return embeddings

