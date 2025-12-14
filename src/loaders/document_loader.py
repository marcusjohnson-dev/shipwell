import os
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)


class DocumentLoader:
    """Loads documents from different formats into LangChain Doc objects."""
    
    SUPPORTED_FORMATS = {
        '.txt': 'text',
        '.pdf': 'pdf',
        '.docx': 'docx',
    }
    
    def load_directory(self, folder_path: str) -> List[Document]:
        """Load all docs fro the input folder(sample_docs)."""
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        all_documents = []
        
        for file_path in folder.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                
                if ext in self.SUPPORTED_FORMATS:
                    print(f"Loading: {file_path.name}")
                    docs = self._load_file(str(file_path), ext)
                    all_documents.extend(docs)
        
        print(f"\nLoaded {len(all_documents)} documents")
        return all_documents
    
    def _load_file(self, file_path: str, extension: str) -> List[Document]:
        """Load a single file based on its extension."""
        if extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        
        elif extension == '.pdf':
            loader = PyPDFLoader(file_path)
            return loader.load()

        elif extension == '.docx':
            loader = Docx2txtLoader(file_path)
            return loader.load()
        
        return []