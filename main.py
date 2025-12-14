import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import tempfile

# Load environment variables
load_dotenv()

# Global variables to store the retriever and chat history
vectorstore = None
retriever = None

def process_pdf(pdf_file):
    """Process the uploaded PDF and create a RAG chain"""
    global vectorstore, retriever
    
    if pdf_file is None:
        return "Please upload a PDF file first."
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file)
            tmp_path = tmp_file.name
        
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        # Build FAISS in-memory vector store (Chroma removed due to hnswlib build issues on Python 3.13)
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return f"✅ PDF processed successfully! Found {len(documents)} pages. You can now ask questions about the document."
        
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"

def chat(message, qa_history):
    """Generate assistant reply given user message and prior QA history.

    qa_history: list of tuples [(user_msg, assistant_reply), ...] where assistant_reply can be None for the pending turn.
    """
    global retriever

    if retriever is None:
        return "Please upload and process a PDF file first."

    try:
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        
        # Retrieve relevant documents
        docs = retriever.invoke(message)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Build messages list manually
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on the provided context. "
                      "Use the following pieces of context to answer the question. "
                      "If you don't know the answer, just say you don't know."
        })
        
        # Add chat history
        for user_msg, assistant_msg in qa_history:
            if assistant_msg is not None:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add current question with context
        messages.append({
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {message}"
        })

        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"❌ Error: {str(e)}"


def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="PDF RAG Chatbot") as demo:
        gr.Markdown(
            """
            # 📚 PDF RAG Chatbot
            Upload a PDF document and ask questions about its content using OpenAI's GPT models.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="binary"
                )
                process_btn = gr.Button("Process PDF", variant="primary")
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )

            with gr.Column(scale=2):
                gr.Markdown("### Chat with your PDF")
                chatbot_display = gr.Chatbot(
                    label="Conversation",
                    height=400
                )
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="Ask a question about the document...",
                    lines=2
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")

                gr.Examples(
                    examples=[
                        "What is this document about?",
                        "Can you summarize the main points?",
                        "What are the key findings?"
                    ],
                    inputs=msg
                )

        process_btn.click(
            fn=process_pdf,
            inputs=[pdf_input],
            outputs=[status_output]
        )

        def user_submit(message, history):
            if not message or message.strip() == "":
                return message, history
            if history is None:
                history = []
            return "", history + [[message, None]]

        def bot_response(history):
            if history is None or len(history) == 0:
                return history
            user_message, assistant_message = history[-1]
            if assistant_message is None:
                reply = chat(user_message, history[:-1])
                history[-1] = [user_message, reply]
            return history

        msg.submit(user_submit, [msg, chatbot_display], [msg, chatbot_display]).then(
            bot_response, chatbot_display, chatbot_display
        )
        submit_btn.click(user_submit, [msg, chatbot_display], [msg, chatbot_display]).then(
            bot_response, chatbot_display, chatbot_display
        )
        clear_btn.click(lambda: [], None, chatbot_display, queue=False)

        gr.Markdown(
            """
            ---
            **Note:** Make sure to set your `OPENAI_API_KEY` in the `.env` file.
            """
        )

    return demo

def main():
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(share=False)

if __name__ == "__main__":
    main()
