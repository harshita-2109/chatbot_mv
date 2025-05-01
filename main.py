import os
import warnings
import logging
import base64
from dotenv import load_dotenv  
import streamlit as st
import groq
import tempfile
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage

# Phase 2 libraries
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Phase 3 libraries
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()

# ========== 🖼️ Background Image with Overlay ==========

def add_bg_with_overlay(image_file_path):
    with open(image_file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.6)),
                              url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }}
        
        /* Custom CSS for beautiful affirmation and meditation cards */
        .affirmation-card, .meditation-card {{
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #6c5ce7;
        }}
        
        .affirmation-card {{
            border-left: 5px solid #6c5ce7;
        }}
        
        .meditation-card {{
            border-left: 5px solid #00cec9;
        }}
        
        .card-title {{
            color: #2d3436;
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 15px;
        }}
        
        .card-content {{
            color: #2d3436;
            font-size: 1.1rem;
            line-height: 1.6;
            font-style: italic;
        }}
        
        .card-footer {{
            color: #636e72;
            font-size: 0.9rem;
            margin-top: 15px;
        }}
        
        /* Button styling */
        .stButton>button {{
            background-color: #6c5ce7;
            color: white;
            border-radius: 30px;
            padding: 10px 25px;
            font-weight: 500;
            border: none;
            transition: all 0.3s ease;
            width: 100%;
        }}
        
        .stButton>button:hover {{
            background-color: #5649c0;
            box-shadow: 0 5px 15px rgba(108, 92, 231, 0.3);
            transform: translateY(-2px);
        }}
        
        /* Chat input styling */
        .stTextInput>div>div>input {{
            border-radius: 30px;
            border: 2px solid #dfe6e9;
            padding: 10px 20px;
        }}
        
        /* Main title styling */
        h1 {{
            color: #2d3436;
            font-weight: 700;
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Call it with your image path
add_bg_with_overlay("bg.png")  

# ========== 🔐 API Key Handling ==========

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ ERROR: GROQ_API_KEY is missing! Add it to the .env file.")
    st.stop()

# Suppress warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ========== 💬 Chatbot UI ==========

st.title('Ask ManoVaani!')

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'affirmation' not in st.session_state:
    st.session_state.affirmation = None

if 'meditation' not in st.session_state:
    st.session_state.meditation = None

# Function to detect if the input is related to mental health or greetings
def is_mental_health_query(user_input):
    user_input = user_input.lower()

    # ✅ Allowed greetings
    greeting_keywords = ["hi", "hello", "hey", "hii", "how are you", "what's up", "good morning", "good evening"]
    if any(greet in user_input for greet in greeting_keywords):
        return True

    # ✅ Mental health keywords
    mental_health_keywords = [
        "anxiety", "depression", "mental health", "stress", "therapy", "therapist", "mindfulness",
        "feeling", "emotion", "mood", "panic", "cope", "relax", "overwhelmed", "self-care",
        "mental", "burnout", "fear", "sad", "happy", "lonely", "tired", "hopeless", "healing",
        "confidence", "motivation", "self esteem", "suicidal", "trauma", "grief", "breathe"
    ]

    return any(keyword in user_input for keyword in mental_health_keywords)

# Display all previous messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# ========== 📄 PDF Vector Store with FAISS ==========

def create_vectorstore(pdf_path, db_path):
    """Create a FAISS vectorstore from PDF and save to disk"""
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
    
    # Create and save FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(db_path)
    
    return vectorstore

@st.cache_resource
def get_vectorstore():
    """Get the FAISS vectorstore - create it if it doesn't exist"""
    pdf_path = "./BMSL.pdf"
    db_path = "./faiss_index"
    
    # Check if the vectorstore already exists
    if not os.path.exists(db_path):
        with st.spinner("Setting up knowledge base..."):
            vectorstore = create_vectorstore(pdf_path, db_path)
    else:
        # Load existing vectorstore
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
        vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    return vectorstore

# Admin-only database recreation functionality (hidden from regular users)
# To use this, you would need to create an admin route or special access mechanism
if 'admin_mode' in st.session_state and st.session_state.admin_mode:
    if st.sidebar.button("Recreate Vector Database"):
        st.session_state.recreate_db = True
        
    # Check if we need to recreate the database
    if st.session_state.get('recreate_db', False):
        # Remove the cached resource to force recreation
        try:
            # Clear the cache by using a side effect
            db_path = "./faiss_index"
            if os.path.exists(db_path):
                # Create a temporary directory and move the files there
                # This ensures we don't delete files while they might be in use
                temp_dir = tempfile.mkdtemp()
                shutil.move(db_path, temp_dir)
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Force recalculation on next access
            st.cache_resource.clear()
        except Exception as e:
            pass
        
        # Reset the flag
        st.session_state.recreate_db = False

# ========== 🌟 Affirmation and Meditation Features ==========
def generate_affirmation():
    prompt = "Provide a positive affirmation to encourage someone who is feeling stressed or overwhelmed. Keep it concise, positive, and powerful."
    response = groq.Groq(api_key=api_key).chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_meditation_guide():
    prompt = "Provide a 5-minute guided meditation script to help someone relax and reduce stress. Keep it simple and effective."
    response = groq.Groq(api_key=api_key).chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ========== 🎨 UI Layout with columns ==========
st.markdown("### Mental Wellness Tools")
col1, col2 = st.columns(2)

with col1:
    if st.button("Give me a positive Affirmation"):
        with st.spinner("Generating your affirmation..."):
            st.session_state.affirmation = generate_affirmation()

with col2:
    if st.button("Give me a guided meditation"):
        with st.spinner("Creating your meditation guide..."):
            st.session_state.meditation = generate_meditation_guide()

# Display affirmation in a beautiful card if available
if st.session_state.affirmation:
    st.markdown(f"""
    <div class="affirmation-card">
        <div class="card-title">✨ Your Affirmation</div>
        <div class="card-content">"{st.session_state.affirmation}"</div>
        <div class="card-footer">Repeat this affirmation to yourself whenever you need strength and encouragement.</div>
    </div>
    """, unsafe_allow_html=True)

# Display meditation in a beautiful card if available
if st.session_state.meditation:
    st.markdown(f"""
    <div class="meditation-card">
        <div class="card-title">🧘 Your Guided Meditation</div>
        <div class="card-content">{st.session_state.meditation}</div>
        <div class="card-footer">Find a quiet space, breathe deeply, and follow this guide at your own pace.</div>
    </div>
    """, unsafe_allow_html=True)

# ========== 📝 Prompt Input ==========
prompt = st.chat_input('Share whatever is on your mind...')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    if is_mental_health_query(prompt):
        groq_sys_prompt = ChatPromptTemplate.from_template(
            """You are very smart at everything. Answer the following Question: {user_prompt}.
               Start the answer directly. No small talk please."""
        )

        model = "llama3-8b-8192"
        groq_chat = ChatGroq(
            groq_api_key=api_key,
            model_name=model
        )

        try:
            with st.spinner("ManoVaani is thinking..."):
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("⚠️ Failed to load document")
                    st.stop()
                
                chain = RetrievalQA.from_chain_type(
                    llm=groq_chat,
                    chain_type='stuff',
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True
                )

                result = chain({"query": prompt})
                response = result["result"]
                st.chat_message('assistant').markdown(response)
                st.session_state.messages.append({'role': 'assistant', 'content': response})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    else:
        st.chat_message('assistant').markdown("Sorry, I can only answer questions related to mental health and well-being.")
        st.session_state.messages.append({'role': 'assistant', 'content': "Sorry, I can only answer questions related to mental health and well-being."})

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; color: #636e72; font-size: 0.8rem;">
    ManoVaani - Your Mental Health Companion
</div>
""", unsafe_allow_html=True)
