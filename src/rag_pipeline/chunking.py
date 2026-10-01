from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from .documents_load import load_documents

def chunk_files(docs: List[Document]) -> str:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = True,
        keep_separator = ["\n\n","\n"," ",""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Total number of chunks created {len(chunks)}")
    return chunks

# if __name__ == "__main__":
#     docs = load_documents("resources")
#     chunks = chunk_files(docs)
#     print(f"Total number of chunks created len(chunks)")