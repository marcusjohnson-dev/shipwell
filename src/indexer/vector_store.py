from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


class VectorStoreManager:
    """Manages document chunking, embedding, and vector storage."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.embeddings = OpenAIEmbeddings()
        self.persist_directory = persist_directory
        self.vectorstore = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def index_documents(self, documents: List[Document]) -> None:
        """Chunk documents and add them to the vector store."""
        print("Chunking documents...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        print("Creating embeddings and indexing...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("Indexing complete!")
    
    def get_retriever(self, k: int = 3):
        """Get a retriever for querying the vector store."""
        if self.vectorstore is None:
            raise ValueError("No documents indexed yet!")
        
        return self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )