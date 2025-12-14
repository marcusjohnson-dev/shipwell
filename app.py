import argparse
import os
from dotenv import load_dotenv
from src.loaders import DocumentLoader
from src.indexer import VectorStoreManager
from src.agent import QAAgent


def main():
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in .env file")
        return
    
    parser = argparse.ArgumentParser(description="RAG Agent")
    parser.add_argument("--docs", required=True, help="Path to documents folder")
    args = parser.parse_args()
    
    print("Loading documents...")
    loader = DocumentLoader()
    documents = loader.load_directory(args.docs)
    
    if not documents:
        print("No documents found!")
        return
    
    print("Indexing documents...")
    vectorstore = VectorStoreManager()
    vectorstore.index_documents(documents)
    
    print("Starting Q&A session")
    agent = QAAgent(vectorstore)
    
    print("\nReady! Type 'quit' to exit.\n")
    
    while True:
        question = input("Question: ").strip()
        
        if question.lower() in ['quit', 'exit']:
            break
        
        if not question:
            continue
        
        response = agent.ask(question)
        print(f"\nAnswer: {response['answer']}\n")


if __name__ == "__main__":
    main()