import streamlit as st
import os
import tempfile
from pathlib import Path
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import spacy
import os

import spacy
from spacy.cli import download

def load_spacy_model(model_name):
    try:
        return spacy.load(model_name)
    except OSError:
        download(model_name)
        return spacy.load(model_name)

# Load the model safely
nlp = load_spacy_model("en_core_web_md")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import SpacyEmbeddings


# App title and configuration
st.set_page_config(page_title="Educational RAG App", page_icon="📚", layout="wide")

# App title
st.title("📚 Smart Study Assistant")
st.write("Upload your textbooks and get personalized learning content")

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
    
# Pre-set the API key (hardcoded)
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = "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"

# Sidebar for model selection and preferences
with st.sidebar:
    st.header("Configuration")
    # Set API key as already configured
    st.success("API Key: ✓ Pre-configured")
    
    model_name = st.selectbox(
        "Select Groq Model:",
        ["llama3-70b-8192", "llama3-8b-8192"],
        index=0
    )
    
    # Show uploaded files
    st.header("Uploaded Textbooks")
    if st.session_state.uploaded_files:
        for file in st.session_state.uploaded_files:
            st.write(f"- {file}")
    else:
        st.write("No textbooks uploaded yet.")
    
    # Add info about Spacy models
    st.header("Embedding Configuration")
    spacy_model = st.selectbox(
        "Select Spacy Model:",
        ["en_core_web_md", "en_core_web_lg"],
        index=0,
        help="Medium (md) or Large (lg) Spacy model for embeddings"
    )
    
    # Learning style preferences
    st.header("Learning Preferences")
    learning_style = st.selectbox(
        "Learning Style:",
        ["Visual", "Auditory", "Read/Write", "Kinesthetic", "Balanced"],
        index=4,
        help="Your preferred way of learning new information"
    )
    
    complexity_level = st.select_slider(
        "Content Complexity:",
        options=["Beginner", "Intermediate", "Advanced", "Expert"],
        value="Intermediate",
        help="Adjust the complexity level of the answers"
    )
    
    include_examples = st.checkbox("Include examples in answers", value=True)
    include_analogies = st.checkbox("Include analogies in answers", value=True)
    include_questions = st.checkbox("Include practice questions", value=True)

# Check if Spacy model is installed, if not show instructions
def check_spacy_model(model_name):
    try:
        import spacy
        spacy.load(model_name)
        return True
    except:
        return False

if not check_spacy_model(spacy_model):
    st.warning(f"Spacy model '{spacy_model}' not found. Please install it with:")
    st.code(f"python -m spacy download {spacy_model}")

# Function to process PDFs and create vectorstore
def process_pdfs(pdf_files):
    temp_dir = tempfile.mkdtemp()
    all_docs = []
    uploaded_filenames = []
    
    with st.spinner("Processing PDFs and creating vector embeddings..."):
        # Save uploaded files to temp directory and load them
        for pdf in pdf_files:
            temp_pdf_path = os.path.join(temp_dir, pdf.name)
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf.getbuffer())
            
            try:
                loader = PyPDFLoader(temp_pdf_path)
                docs = loader.load()
                if docs:
                    all_docs.extend(docs)
                    uploaded_filenames.append(pdf.name)
                    st.success(f"Successfully processed: {pdf.name}")
                else:
                    st.warning(f"No content extracted from: {pdf.name}")
            except Exception as e:
                st.error(f"Error processing {pdf.name}: {str(e)}")
    
    if all_docs:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(all_docs)
        
        # Create vector store using Spacy embeddings
        try:
            # Initialize Spacy embeddings
            embeddings = SpacyEmbeddings(model_name=spacy_model)
            vectorstore = FAISS.from_documents(splits, embeddings)
            st.success(f"✅ Created embeddings for {len(splits)} text chunks using Spacy {spacy_model}")
            return vectorstore, uploaded_filenames
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return None, []
    else:
        st.error("No documents were successfully processed")
        return None, []

# File uploader
uploaded_pdfs = st.file_uploader("Upload PDF Textbooks", type="pdf", accept_multiple_files=True)

# Process PDFs when user clicks the button
if uploaded_pdfs and st.button("Process Textbooks"):
    if check_spacy_model(spacy_model):
        vectorstore, filenames = process_pdfs(uploaded_pdfs)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.uploaded_files.extend([name for name in filenames if name not in st.session_state.uploaded_files])
            st.success("Textbooks processed and ready for questions!")
    else:
        st.error(f"Please install the required Spacy model first: python -m spacy download {spacy_model}")

# Create the RAG chain
def create_rag_chain():
    # Set environment variable (the API key is already in session state)
    os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key
    
    # Create retriever
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Create LLM
    llm = ChatGroq(model_name=model_name)
    
    # Get learning preferences from sidebar
    learning_style = st.session_state.get("learning_style", "Balanced")
    complexity = st.session_state.get("complexity_level", "Intermediate")
    include_examples = st.session_state.get("include_examples", True)
    include_analogies = st.session_state.get("include_analogies", True)
    include_questions = st.session_state.get("include_questions", True)
    
    # Create enhanced prompt template for better student learning
    template = """
    You are an expert educational assistant helping students learn effectively.
    
    The student has the following learning preferences:
    - Learning style: {learning_style}
    - Content complexity level: {complexity}
    - Include examples: {include_examples}
    - Include analogies: {include_analogies}
    - Include practice questions: {include_questions}
    
    Use the following context from textbooks to answer the question:
    {context}
    
    Question: {question}
    
    Format your answer for optimal student learning:
    
    1. Start with a clear, concise summary of the main concept (2-3 sentences)
    
    2. Present key points in well-structured bullet points:
       • Use bold for important terms or concepts
       • Keep each bullet focused on a single idea
       • Provide context and importance for each point
       • Use sub-bullets for supporting details
    
    3. If examples are requested, provide relevant examples that make the concept concrete
    
    4. If analogies are requested, use analogies to connect new information to familiar concepts
    
    5. Organize information with clear headings (## Heading) for different sections
    
    6. Use markdown formatting to make your answer visually clear and organized
    
    7. If practice questions are requested, include 2-3 practice questions at the end with answers
    
    8. Always cite which parts of the textbook you're referencing
    
    Remember to adapt the complexity to the student's level and use their preferred learning style in your explanations. For visual learners, describe in spatial terms. For auditory learners, emphasize dialogue and sound. For read/write learners, focus on definitions and lists. For kinesthetic learners, use action-oriented examples.
    
    Your Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the chain
    chain = (
        {"context": retriever, 
         "question": RunnablePassthrough(),
         "learning_style": lambda _: learning_style,
         "complexity": lambda _: complexity,
         "include_examples": lambda _: str(include_examples),
         "include_analogies": lambda _: str(include_analogies),
         "include_questions": lambda _: str(include_questions)
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# Question answering section
st.header("Learn From Your Textbooks")

# UI for different learning modes
tab1, tab2, tab3 = st.tabs(["Ask Questions", "Study Guide Generator", "Concept Explorer"])

with tab1:
    # Check if vectorstore is ready
    if st.session_state.vectorstore is None:
        st.info("Please upload and process textbooks before asking questions.")
    else:
        # Get user question
        question = st.text_input("What would you like to learn about?")
        
        if question and st.button("Get Answer", key="answer_btn"):
            try:
                # Create the chain
                rag_chain = create_rag_chain()
                
                if rag_chain:
                    with st.spinner("Finding the best answer for you..."):
                        response = rag_chain.invoke(question)
                    
                    # Display the answer
                    st.markdown(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("This might be due to a connection issue with Groq.")

with tab2:
    if st.session_state.vectorstore is None:
        st.info("Please upload and process textbooks first.")
    else:
        st.write("Generate a comprehensive study guide on any topic from your textbooks.")
        topic = st.text_input("Topic for study guide:")
        if topic and st.button("Generate Study Guide", key="guide_btn"):
            try:
                rag_chain = create_rag_chain()
                if rag_chain:
                    with st.spinner("Creating your personalized study guide..."):
                        study_prompt = f"Create a comprehensive study guide about {topic}. Include key definitions, important concepts, relationships between ideas, and a summary of the main points."
                        response = rag_chain.invoke(study_prompt)
                    st.markdown(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab3:
    if st.session_state.vectorstore is None:
        st.info("Please upload and process textbooks first.")
    else:
        st.write("Explore connections between concepts in your textbooks.")
        concept1 = st.text_input("First concept:")
        concept2 = st.text_input("Second concept:")
        if concept1 and concept2 and st.button("Explore Connection", key="explore_btn"):
            try:
                rag_chain = create_rag_chain()
                if rag_chain:
                    with st.spinner("Exploring connections..."):
                        connect_prompt = f"Explain the relationship between {concept1} and {concept2}. How are they connected? What are the similarities and differences? How do they interact or influence each other?"
                        response = rag_chain.invoke(connect_prompt)
                    st.markdown(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Reset button to clear the vectorstore and uploaded files
if st.button("Reset App"):
    st.session_state.vectorstore = None
    st.session_state.uploaded_files = []
    st.success("App reset successfully. You can upload new textbooks now.")

# Footer
st.markdown("---")
st.caption("Smart Study Assistant | Built with Streamlit, Langchain, Groq API, and Spacy Embeddings")
