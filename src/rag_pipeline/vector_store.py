import chromadb
from .chunking import chunk_files
from .documents_load import load_documents
from .embeddings import generate_embeddings

class store_in_db:
    def __init__(self,name,persist_dir):
        self.name = name
        self.persist_dir = persist_dir

    def create_db(self):
        self.client = chromadb.PersistentClient(self.persist_dir)
        self.collection = self.client.get_or_create_collection(self.name)
        count = self.collection.count()
        print(f"Number of existing documents in a collection before adding {count}")

    def add_to_db(self,chunks,embeddings):
        documents = [chunk.page_content for chunk in chunks]
        ids = [f"chunk_id {i}" for i in range(len(chunks))]
        self.collection.add(
            ids,
            embeddings,
            metadatas = [
                {"source":ids[i]} for i in range(len(chunks))
            ],
            documents = documents
            )
        count = self.collection.count()
        print(f"Number of existing documents in a collection after adding documents {count}")


if __name__ == "__main__":
    docs = load_documents("resources")
    chunks = chunk_files(docs)
    embeddings = generate_embeddings(chunks)
    client_obj = store_in_db("ashritha_vector_store",r"C:\Users\Ashritha\Downloads\rag-pipeline\src\rag_pipeline\vector_store")
    client_obj.create_db()
    client_obj.add_to_db(chunks,embeddings)
