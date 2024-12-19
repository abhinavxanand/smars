import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import ffmpeg
import torch
import tempfile
import os
from autogen import register_function, AssistantAgent, UserProxyAgent
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
from transformers import pipeline
from openai import AzureOpenAI
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Azure OpenAI setup
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_API_BASE = os.getenv("AZURE_API_BASE")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

az_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_API_BASE,
    api_version=AZURE_API_VERSION
)

# Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Tavily Client
tavily_client = TavilyClient(api_key="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")

# Whisper pipeline
whisper = pipeline('automatic-speech-recognition', model='openai/whisper-medium',device = "cuda" if torch.cuda.is_available() else "cpu")

# Multi-Agent Framework Setup
ROUTER_PROMPT = """
You are an intelligent query routing system tasked with first extracting the questions from query and then selecting the best method to answer a user’s query. The methods available are:

1. ** Call RAG tool **: Use when the query requires answering based on pre-existing documents (knowledge base) that is Resume/CV of the user which contains his/her certifications, education, projects and other relevant details, so if the query seems personal question about user then use RAG.

2. **Call Websearch tool**: Use when the query involves topics that are likely to require up-to-date information, external knowledge, or content not available in the knowledge base. This method will involve querying a web search API like Tavily and summarizing the results.

If none of above tool are suitable then answer yourself : Do when the query is highly specific and requires inference, reasoning, or generating creative content. You can provide answers based purely on your training without requiring retrieval from external documents.

### **Rules for Deciding the Method**:

- If the query asks about **historical information**, **fact-based knowledge** from pre-existing sources, or if the answer can be found in the available document corpus, use **RAG**.
- If the query is related to **up-to-date information** or external data that is not available in the knowledge base, like news, trends, or specific external sources, route the query to **Websearch**.
- If the query is **open-ended**, involves **creativity**, requires **reasoning**, or is **highly specific** to a domain not covered in the available documents or through web search, answer it yourself.

### **Example Scenarios**:

1. **Query**: "What is the plot of The Dark Knight movie?"
   - **Decision**: Answer yourself. The query requires a synthesis of general knowledge and a creative summary, which is best handled by an LLM.

2. **Query**: "What was the weather in New York City on June 1st, 2023?"
   - **Decision**: Use **Websearch**. The query involves specific historical data that can be retrieved via a web search API for accurate results.

3. **Query**: "What is meant by Gotham needs me?"
   - **Decision**: Use **RAG**. The query pertains to information likely to be available in the existing document corpus or knowledge base that is Batman Dark Knight script.

4. **Query**: "How does the new AI model improve upon previous versions?"
   - **Decision**: Use **Websearch**. The query relates to a recent development, and web search will retrieve the most current information.

5. **Query**: "Tell me about the history of artificial intelligence."
   - **Decision**: Answer yourself. The query involves historical data, which is likely to be answered by yourself.

### **Response Format**:
- **If RAG tool is called**:
  "I will retrieve the most relevant information from our knowledge base and generate an answer based on it."

- **If Websearch tool is called**:
  "I will search the web for the most up-to-date information on this topic and summarize the results."

- **If Answering yourself**:
  "I will use my language model capabilities to generate an answer based on my knowledge."

Please use these guidelines to decide the appropriate method based on the user's query and provide the necessary output accordingly.

"""
llm_config = {
    "model": AZURE_DEPLOYMENT_NAME,
    "api_key": AZURE_API_KEY,
    "base_url": AZURE_API_BASE,
    "api_type": "azure",
    "api_version": AZURE_API_VERSION
}

LLM_Router = AssistantAgent(
    name="LLM_Router",
    code_execution_config={"use_docker": False},
    system_message=ROUTER_PROMPT,
    max_consecutive_auto_reply=1,
    llm_config=llm_config,
)

Admin = UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the LLM_Router to get the answer generated. Show the generated answer to the user.",
    max_consecutive_auto_reply=1,
    code_execution_config={"use_docker": False},
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: isinstance(msg, dict) and msg.get("content") and "Answer:" in msg["content"],
)

# Helper Functions
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_text_chunks(chunks):
    return model.encode(chunks)

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def process_audio(file_path):
    return whisper(file_path, return_timestamps=True)

def rag_query(question, index, chunks, k=3):
    query_embedding = model.encode([question])
    distances, indices = index.search(query_embedding.astype("float32"), k=k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return " ".join(retrieved_chunks)

def generate_answer(question, context):
    rag_prompt = f"""
    You will be given two contexts to consider:
    One is Question and other is the reference for answering i.e. the chunks retrieved from the RAG (Retrieval-Augmented Generation) system, which contain information relevant to the user's query
    Question: {question}
    Context: {context}
    Your task is to give the most appropriate answer to the question, based on the retrieved chunks i.e. Context.
    Remember the generated answer should start with ** Answer: **
    """
    response = az_client.chat.completions.create(
        model="ak-gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": rag_prompt},
            {"role": "user", "content": f"Question: {question}\nContext: {context}"}
        ]
    )
    return response.choices[0].message.content

def RAG(question: str) -> str:
    context = rag_query(question, faiss_index, text_chunks)
    answer = generate_answer(question, context)
    return answer

def WebSearch(question: str):
    search_context = tavily_client.get_search_context(query=question)
    answer = generate_answer(question, search_context)
    return answer

# Register Functions
register_function(
    RAG,
    caller=LLM_Router,
    executor=Admin,
    name="RAG_Tool",
    description="RAG-based answer generation."
)
register_function(
    WebSearch,
    caller=LLM_Router,
    executor=Admin,
    name="WebSearch",
    description="Web search-based answer retrieval and summarization."
)

import streamlit as st
import pyaudio
import wave
from pydub import AudioSegment
import tempfile
import numpy as np

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.data = []
        self.is_recording = False
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.channels = 1

    def start_recording(self):
        self.is_recording = True
        self.data = []
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.callback
        )
        self.stream.start_stream()

    def stop_recording(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.is_recording = False

    def callback(self, in_data, frame_count, time_info, status):
        self.data.append(in_data)
        return (in_data, pyaudio.paContinue)

    def get_audio_data(self):
        return b''.join(self.data)

# Streamlit UI
st.title("Multi-Agent Framework with RAG System and Microphone Recording")
response = ""
# Step 1: Upload PDF
uploaded_pdf = st.file_uploader("Step 1: Upload a PDF file", type="pdf")
wav_file_path = ""
mp3_file_path = ""
if uploaded_pdf:
    st.info("Processing PDF...")

    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    pdf_text = extract_text_from_pdf("uploaded_pdf.pdf")
    text_chunks = chunk_text(pdf_text)
    embeddings = embed_text_chunks(text_chunks)
    faiss_index = create_faiss_index(embeddings)
    st.success("PDF processed and indexed!")

    # Step 2: Microphone Input for Audio Recording
    st.write("Step 2: Record audio using the microphone")

    # Initialize recorder in session state
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()
        st.session_state.recording_status = False
        st.session_state.audio_processed = False

    # Recording controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('Start Recording 🎙️'):
            st.session_state.recorder.start_recording()
            st.session_state.recording_status = True
            st.session_state.audio_processed = False
            
    with col2:
        if st.button('Stop Recording 🛑'):
            if st.session_state.recording_status:
                st.session_state.recorder.stop_recording()
                st.session_state.recording_status = False
                
                wav_file_path = "audio_output.wav"
                mp3_file_path = "audio_output.mp3"

                # Save audio to temporary WAV file
                temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                with wave.open(wav_file_path, 'wb') as wf:
                    wf.setnchannels(st.session_state.recorder.channels)
                    wf.setsampwidth(st.session_state.recorder.audio.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(st.session_state.recorder.sample_rate)
                    wf.writeframes(st.session_state.recorder.get_audio_data())

                audio = AudioSegment.from_wav(wav_file_path)
                audio.export(mp3_file_path, format="mp3")
                # Display audio player
                with open(mp3_file_path, 'rb') as f:
                    st.audio(f.read(), format='audio/wav')
    

                # Process the audio
                if not st.session_state.audio_processed:
                    st.success("Recording complete! Processing audio...")

                    # Display Transcription
                    st.write("**Transcription:**")
                    text = whisper(mp3_file_path, return_timestamps=True, generate_kwargs={"language": "en"})
                    transcription = text['text']
                    st.write(transcription)

                    # Process Transcription as a Question
                    st.write("**Question Identified:**")
                    user_question = transcription
                    st.write(user_question)

                    # LLM Decision
                    st.write("**LLM Decision:**")
                    decision = "Using ____ tool."  # Example decision logic
                    st.write(decision)

                    # Get the answer from the framework
                    st.write("**Answer:**")
                    Admin.initiate_chat(LLM_Router, message=user_question)
                    response = LLM_Router.last_message()["content"]
                    # st.write(response)

                    st.session_state.audio_processed = True
    st.write(response)
    # Display recording status
    if st.session_state.recording_status:
        st.write("🎙️ Recording in progress...")
