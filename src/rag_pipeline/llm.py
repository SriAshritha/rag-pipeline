from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from .documents_load import load_documents
from .chunking import chunk_files
from .embeddings import generate_embeddings
from .vector_store import store_in_db

import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

class retrieval():

    def __init__(self,user_query,top_k):
        self.query = user_query
        self.top_k = top_k

    def encode_query(self):
        load_model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = load_model.encode(self.query,show_progress_bar = True)
        print(f"Successfully encoded the query with dimension {len(query_embedding)}")

        return query_embedding

    def retrieve(self, query_embedding, vector_db_collection):
        results = vector_db_collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )
        docs = results["documents"][0]

        return docs

    def re_ranking(self,docs):
        ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [ (self.query , doc) for doc in docs]
        scores = ranker.predict(pairs)

        ranked_docs = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        for doc, score in ranked_docs:
            print(f"\nScore: {score}")
            print(doc)

        return ranked_docs

class PromptToLLM():
    def __init__(self,ranked_docs,user_query):
        self.system_prompt = "You are CTS HR manager and recruiter.Your Job is to explain and tell the students about the skills required and the hiring process."
        self.context = "\n\n".join(
            doc for doc, score in ranked_docs
        )
        self.query = user_query

    def prompt_creation(self):
        prompt = self.system_prompt + self.context + self.query
        return prompt

    def call_to_llm(self,prompt):
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return response

if __name__ == "__main__":
    
    docs = load_documents("resources")
    chunks = chunk_files(docs)
    embeddings = generate_embeddings(chunks)
    client_obj = store_in_db("ashritha_vector_store",r"C:\Users\Ashritha\Downloads\rag-pipeline\src\rag_pipeline\vector_store")
    client_obj.create_db()
    client_obj.add_to_db(chunks,embeddings)

    user_query = "What are the skills required to crack CTS Ace Engineer role ?"
    obj = retrieval(user_query,3)
    encoded_query = obj.encode_query()
    docs = obj.retrieve(encoded_query,client_obj.collection)
    ranked_docs = obj.re_ranking(docs)

    client = PromptToLLM(ranked_docs,user_query)
    prompt = client.prompt_creation()
    response = client.call_to_llm(prompt)

    print(response.text)

    
    

