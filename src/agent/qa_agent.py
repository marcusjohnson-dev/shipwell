from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


class QAAgent:
    """Agent for answering questions using retrieved context."""
    
    def __init__(self, vectorstore_manager):
        self.retriever = vectorstore_manager.get_retriever(k=3)
        
        # Using GPT-3.5 for cost efficiency
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
    
    def ask(self, question: str) -> dict:
        """Ask a question and get an answer with sources."""
        # I used .invoke() here instead of call to fix deprecation warning
        result = self.chain.invoke({"query": question})
        
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }