import argparse
from dotenv import load_dotenv
from src.loaders import DocumentLoader
from src.indexer import VectorStoreManager
from src.agent import QAAgent

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Used this to disable ChromaDB telemetry; still didn't work, might need to upgrade DB


def main():
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: API_KEY not found in env")
        return
    
    parser = argparse.ArgumentParser(description="RAG Agent")
    parser.add_argument("--docs", required=True, help="Path to documents folder")
    args = parser.parse_args()
    
    print("=" * 60)
    print("RAG AGENT - Document Q&A System")
    print("=" * 60)
    
    # Step 1: Load documents
    print(f"\n[1/3] Loading documents from: {args.docs}")
    loader = DocumentLoader()
    try:
        documents = loader.load_directory(args.docs)
        if not documents:
            print("No documents found! Please add supported files to the folder.")
            return
    except Exception as e:
        print(f"Error loading documents: {e}")
        return
    
    # Step 2: Index documents
    print(f"\n[2/3] Indexing documents...")
    vectorstore = VectorStoreManager()
    vectorstore.index_documents(documents)
    
    # Step 3: Start Q&A loop
    print(f"\n[3/3] Starting Q&A session")
    agent = QAAgent(vectorstore)
    
    print("\n" + "=" * 60)
    print("Ready! Ask questions about your documents.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")
    
    while True:
        question = input("\n🤔 Your Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not question:
            continue
        
        try:
            # Get answer
            response = agent.ask(question)
            
            # Display answer
            print(f"\n💡 Answer:\n{response['answer']}")
            
            # Display sources
            print(f"\n📚 Sources:")
            for i, doc in enumerate(response['sources'], 1):
                source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                preview = doc.page_content[:100].replace('\n', ' ')
                print(f"  [{i}] {source_file}")
                print(f"      Preview: {preview}...")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()