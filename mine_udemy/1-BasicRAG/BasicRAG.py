from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models.base import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json

load_dotenv()
persist_dir = './chroma_db'

class SmartPdfProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.text_splitter =  RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function=len,
            separators = ["\n\n", "\n", " ", ""]
        )

    def process_pdf(self,pdf_path):
        pages=[]
        pdf_loader = PyMuPDFLoader(pdf_path)
        pdf_pages = pdf_loader.load()

        for i,page in enumerate(pdf_pages):
            #Create chunks
            page_content = self._clean_text(page.page_content)
            
            chunks = self.text_splitter.create_documents(
                texts=[page_content],
                metadatas=[
                    {
                        **page.metadata,
                        "page": i+1,
                        "total_pages": len(pdf_pages),
                        "chunk_method": "smart_pdf_processot",
                        "char_count": len(page_content)
                    }                    
                ]
            )

            pages.extend(chunks)

        #Create vector Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        #vectors = embeddings.embed_documents([doc.page_content for doc in chunks])

        #for vector in vectors:
            #Store embeddings into Chromadb

        # Delete existing collection before creating new one
        try:
            existing_vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name="rag_collection"
            )
            existing_vectorstore.delete_collection()
            print(f"Deleted existing collection: rag_collection")
        except Exception as e:
            print(f"No existing collection to delete or error: {e}")
        
        #create documents in vector store
        vectorstore = Chroma.from_documents(
            documents=pages, 
            embedding=embeddings, #OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=persist_dir,
            collection_name="rag_collection"
        )

        print(f"vector store created with {vectorstore._collection.count()} vectors")
        print(f"persisted to {persist_dir}")

        #region print chunk contents
        # Print all records from rag_collection
        # all_docs = vectorstore.get()
        # print(f"\n=== All records from rag_collection ===")
        # print(f"Total documents: {len(all_docs['ids'])}")
        # for i, (doc_id, document, metadata) in enumerate(zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas'])):
        #     print(f"\n--- Record {i+1} ---")
        #     print(f"ID: {doc_id}")
        #     print(f"Metadata: {metadata}")
        #     print(f"Content preview: {document[:200]}...")
        #endregion

        #similar_docs_with_rel_scores = vectorstore.similarity_search_with_relevance_scores("what is Embeddings and Softmax ?", k=3)

        #region print
        '''
        if not similar_docs_with_rel_scores:
            print("No similar documents found.")
        else:
            print("\n=== Top similar documents ===")
            for i, (doc, score) in enumerate(similar_docs_with_rel_scores, 1):
                meta = doc.metadata or {}
                header = f"Match {i} | Relevance: {score:.4f}"
                if "page" in meta:
                    header += f" | Page {meta.get('page')}/{meta.get('total_pages', '?')}"
                print(header)

                if meta:
                    print("Metadata:")
                    for k, v in meta.items():
                        print(f"  - {k}: {v}")

                content = (doc.page_content or "").strip()
                max_chars = 1200
                if len(content) > max_chars:
                    content = content[:max_chars] + "..."
                print("Content:")
                print(content)
                print("-" * 80)
        '''
        #endregion
    
        #Create Retriever
        retriever=vectorstore.as_retriever(
            search_kwargs={"k":3} ## Retrieve top 3 relevant chunks
        )

        #Create LLM object
        #llm = init_chat_model(model="openai:gpt-3.5-turbo")
        llm = init_chat_model(model="groq:openai/gpt-oss-120b")


        #region Create Context + Context prompt + Context Chain
        
        # Contextualize question prompt - rephrases question based on chat history
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{question}")
        ])
        
        # Chain to contextualize the question
        contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
        
        # Function to contextualize question or pass through if no history
        def contextualize_question(input_dict):
            result = ""
            if input_dict.get("chat_history"):
                result= contextualize_q_chain.invoke(input_dict)
            else:
                result = input_dict["question"]
            
            return result
            
        #endregion

        #region Create QA text + QA prompt + QA Chain

        #Build QA Prompt with context
        qa_system_prompt = """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise.

            Context: {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{question}")
        ])
        
        # Create context-aware RAG chain using LCEL
        self.rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: self.format_docs(
                    retriever.invoke(contextualize_question(x))
                )
            )
            | qa_prompt
            #| self.print_prompt
            | llm
            | StrOutputParser()
        )

        #endregion
        
        # Initialize chat history
        self.chat_history = []

    # Function to query the modern RAG system
    def query_rag_modern(self, question):
        print(f"\nQuestion: {question}")
        print("-" * 50)
        
        # Using LCEL chain approach - pass as dictionary with chat_history
        result = self.rag_chain.invoke({
            "question": question,
            "chat_history": self.chat_history
        }, config={"verbose": True})
        
        # Update chat history
        self.chat_history.append(("human", question))
        self.chat_history.append(("assistant", result))
        
        print(f"\n=== Response ===")
        #print(result)
        
        return result


    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Fix common PDF extraction issues
        text = text.replace("ﬁ", "fi")
        text = text.replace("ﬂ", "fl")
        
        return text
    
    # Format documents helper function
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Debug chain to print the prompt before LLM
    def print_prompt(self, messages):
        print("\n=== Prompt being sent to LLM ===")
        if hasattr(messages, 'messages'):
            for msg in messages.messages:
                print(f"\n[{msg.__class__.__name__}]")
                print(msg.content)
            else:
                print(messages)
                print("=" * 50 + "\n")
        return messages
    
    
if __name__ == "__main__":
    processor = SmartPdfProcessor(chunk_size=1000, chunk_overlap=100)
    pdf_path = "mine_udemy/0-dataparsing/data/pdf/My Labs.pdf"
    processor.process_pdf(pdf_path)

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            print("Exiting the chat.")
            break
        response = processor.query_rag_modern(question)
        print(f"Assistant: {response}")
