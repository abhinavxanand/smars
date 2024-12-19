# Multi-Agent Framework with RAG System

A Streamlit-based application that combines AutoGen multi-agent framework with RAG (Retrieval Augmented Generation) capabilities, web search, and audio transcription. The system allows users to upload documents, record audio questions, and get intelligent responses through a sophisticated routing system.

## Features

- PDF document processing and indexing using FAISS
- Real-time audio recording and processing
- Speech-to-text transcription using Whisper
- Intelligent query routing between RAG and web search
- Multi-agent framework using AutoGen
- Azure OpenAI integration for language processing
- Web search capabilities using Tavily

## Prerequisites

### API Keys Required
- Azure OpenAI API credentials
- Tavily API key

### Environment Variables
Create a `.env` file with the following variables:
```
AZURE_API_KEY=your_azure_api_key
AZURE_API_VERSION=your_azure_api_version
AZURE_API_BASE=your_azure_endpoint
AZURE_DEPLOYMENT_NAME=your_deployment_name
```

### Required Libraries
```bash
pip install streamlit
pip install streamlit-webrtc
pip install ffmpeg-python
pip install torch
pip install autogen
pip install sentence-transformers
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install PyPDF2
pip install transformers
pip install openai
pip install python-dotenv
pip install tavily-python
pip install pyaudio
pip install pydub
```

## System Architecture

1. **Document Processing Pipeline**
   - PDF text extraction
   - Text chunking
   - Embedding generation
   - FAISS index creation

2. **Audio Processing Pipeline**
   - Real-time audio recording
   - WAV to MP3 conversion
   - Whisper transcription

3. **Multi-Agent Framework**
   - LLM Router: Intelligent query routing
   - Admin Agent: Query execution and response generation
   - RAG Tool: Document-based response generation
   - Web Search Tool: Internet-based response generation

## Usage

1. Start the application:
```bash
streamlit run taskUI2.py
```

2. Upload a PDF document that will serve as the knowledge base.

3. Use the microphone controls to:
   - Start recording your question
   - Stop recording when finished
   - Listen to the recorded audio
   - View the transcribed text
   - Get the AI-generated response

## How It Works

1. **Document Processing**
   - PDF is uploaded and text is extracted
   - Text is chunked and embedded
   - FAISS index is created for efficient retrieval

2. **Query Processing**
   - Audio is recorded and transcribed
   - Query is routed to appropriate tool:
     - RAG: For questions about document content
     - Web Search: For current/external information
     - Direct LLM: For reasoning/creative responses

3. **Response Generation**
   - Context is retrieved from appropriate source
   - Response is generated using Azure OpenAI
   - Answer is displayed to the user

## Component Details

### RAG System
- Uses sentence-transformers for embeddings
- FAISS for similarity search
- Chunking with overlap for context preservation

### Audio Recording
- PyAudio for real-time audio capture
- Wave for audio file handling
- Pydub for audio format conversion

### LLM Router
- Intelligent routing based on query type
- Context-aware decision making
- Multiple response generation methods

## Troubleshooting

1. **Audio Recording Issues**
   - Ensure microphone permissions are enabled
   - Check PyAudio installation
   - Verify audio device is recognized

2. **PDF Processing Issues**
   - Check PDF file format compatibility
   - Ensure sufficient memory for large documents
   - Verify FAISS index creation

3. **API Issues**
   - Verify all environment variables are set
   - Check API key validity
   - Ensure proper Azure OpenAI model deployment

## Contributing

Please submit issues and pull requests for any improvements or bug fixes.

## License

[MIT License](LICENSE)
