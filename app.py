import gradio as gr
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
import logging

# Set up logging for debugging and error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the Hugging Face API token is available
api_token = os.getenv("HF_TOKEN")
if not api_token:
    logger.error("Hugging Face API token (HF_TOKEN) is not set.")
    raise EnvironmentError("Hugging Face API token (HF_TOKEN) is not set.")

# List of available LLMs
list_llm = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]
list_llm_simple = [os.path.basename(llm) for llm in list_llm]

def load_doc(list_file_path):
    """
    Load and split PDF documents into manageable text chunks.
    """
    try:
        loaders = [PyPDFLoader(x) for x in list_file_path]
        pages = []
        for loader in loaders:
            pages.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=64
        )
        doc_splits = text_splitter.split_documents(pages)
        return doc_splits
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise

def create_db(splits):
    """
    Create a vector database from document splits.
    """
    try:
        embeddings = HuggingFaceEmbeddings()
        vectordb = FAISS.from_documents(splits, embeddings)
        return vectordb
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        raise

def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db):
    """
    Initialize the language model chain for conversational retrieval.
    """
    try:
        llm = HuggingFaceEndpoint(
            repo_id=llm_model,
            huggingfacehub_api_token=api_token,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_k=top_k,
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key='answer',
            return_messages=True
        )

        retriever = vector_db.as_retriever()
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            chain_type="stuff",
            memory=memory,
            return_source_documents=True,
            verbose=False,
        )
        return qa_chain
    except Exception as e:
        logger.error(f"Error initializing LLM chain: {e}")
        raise

def initialize_database(list_file_obj):
    """
    Initialize the vector database from uploaded PDF files.
    """
    try:
        list_file_path = [x.name for x in list_file_obj if x is not None]
        if not list_file_path:
            raise ValueError("No valid PDF files uploaded.")
        doc_splits = load_doc(list_file_path)
        vector_db = create_db(doc_splits)
        return vector_db, "Database created successfully!"
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return None, f"Failed to create database: {e}"

def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db):
    """
    Initialize the language model for the chatbot.
    """
    try:
        llm_name = list_llm[llm_option]
        qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db)
        return qa_chain, "QA chain initialized. Chatbot is ready!"
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return None, f"Failed to initialize LLM: {e}"

def format_chat_history(chat_history):
    """
    Format chat history for display.
    """
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history

def conversation(qa_chain, message, history):
    """
    Handle the conversation with the chatbot.
    """
    try:
        formatted_chat_history = format_chat_history(history)
        response = qa_chain.invoke({"question": message, "chat_history": formatted_chat_history})
        response_answer = response["answer"]
        if "Helpful Answer:" in response_answer:
            response_answer = response_answer.split("Helpful Answer:")[-1]
        response_sources = response.get("source_documents", [])
        new_history = history + [(message, response_answer)]
        sources = []
        for i, source in enumerate(response_sources[:3]):
            sources.append((source.page_content.strip(), source.metadata.get("page", 0) + 1))
        return qa_chain, gr.update(value=""), new_history, *sources
    except Exception as e:
        logger.error(f"Error during conversation: {e}")
        return qa_chain, gr.update(value=""), history, ("Error retrieving response.", 0)

def demo():
    """
    Launch the Gradio demo interface.
    """
    with gr.Blocks(theme=gr.themes.Default(primary_hue="red", secondary_hue="pink", neutral_hue="sky")) as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        gr.HTML("<center><h1>RAG PDF Chatbot</h1></center>")
        gr.Markdown("""
        **Query your PDF documents!** This AI agent is designed to perform retrieval-augmented generation (RAG) on PDF documents.
        **Please do not upload confidential documents.**
        """)
        with gr.Row():
            with gr.Column(scale=86):
                gr.Markdown("**Step 1 - Upload PDF documents and Initialize RAG pipeline**")
                with gr.Row():
                    document = gr.Files(height=300, file_count="multiple", file_types=["pdf"], interactive=True, label="Upload PDF documents")
                with gr.Row():
                    db_btn = gr.Button("Create vector database")
                with gr.Row():
                    db_progress = gr.Textbox(value="Not initialized", show_label=False)
                gr.Markdown("**Select Large Language Model (LLM) and input parameters**")
                with gr.Row():
                    llm_btn = gr.Radio(list_llm_simple, label="Available LLMs", value=list_llm_simple[0], type="index")
                with gr.Row():
                    with gr.Accordion("LLM input parameters", open=False):
                        with gr.Row():
                            slider_temperature = gr.Slider(minimum=0.01, maximum=1.0, value=0.5, step=0.1, label="Temperature", 