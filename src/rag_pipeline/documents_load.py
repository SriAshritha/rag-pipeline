import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader

def load_documents(folder_path : str) -> List[Document]:
    """
    Load documents from a given folder path.

    Args:
        folder_path (str): The path to the folder containing the files to be loaded.
    Return type:
        List[Document]: A list of loaded Document objects.
    """
    documents_dir = Path(folder_path)
    if not documents_dir.exists():
        raise FileNotFoundError
    if not documents_dir.is_dir():
        raise NotADirectoryError
    all_docs = []

    for pdf_file in documents_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        all_docs.extend(pages)
        print(f"Loaded {pdf_file.name} with {len(pages)} pages.")

    for excel_file in documents_dir.glob("*.xlsx"):
        loader = UnstructuredExcelLoader(str(excel_file))
        all_docs.extend(loader.load())
        print(f"Loaded excel_file")

    print(f"Total loaded documents: {len(all_docs)}")

    return all_docs

# if __name__ == "__main__":
#     documents = load_documents("../../resources")
#     for i, doc in enumerate(documents):
#         print(f"\n--- Document {i} ---")
#         print("Source:", doc.metadata.get("source"))
#         print("Page:", doc.metadata.get("page"))
#         print("Content length:", len(doc.page_content))
#         print("Content:", doc.page_content[:200])