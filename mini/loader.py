from markitdown import MarkItDown
import os
from typing import List
from mini.logger import logger


class DocumentLoader:
    def __init__(self):
        self.md = MarkItDown()
        
    def load(self, document_path: str):
        result = self.md.convert(document_path)
        return result.text_content

    def load_documents(self, document_paths: List[str]):
        documents = []
        for document_path in document_paths:
            documents.append(self.load(document_path))
        return documents

    def load_documents_from_directory(self, directory_path: str):
        document_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path)]
        return self.load_documents(document_paths)

if __name__ == "__main__":
    loader = DocumentLoader()
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    documents_dir = os.path.join(script_dir, "documents")
    documents = loader.load_documents_from_directory(documents_dir)
    logger.debug(documents)