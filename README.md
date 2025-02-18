# Youtube_Assistant

## 📌 Overview
Youtube_Assistant is an AI-powered Streamlit web application that allows users to extract and analyze YouTube video transcripts using LangChain and FAISS. It enables users to query video content efficiently and receive accurate, AI-generated responses.

## 🚀 Features
- Extracts transcripts from YouTube videos.
- Splits text into manageable chunks for processing.
- Utilizes FAISS for efficient similarity search.
- Leverages Ollama's deep learning model for generating responses.
- Provides a Streamlit-based interactive user interface.

## 🛠️ Tech Stack
- **LangChain** (Text processing, embeddings, FAISS)
- **FAISS** (Efficient similarity search)
- **Sentence-Transformers** (Embedding model)
- **Ollama** (LLM-based response generation)
- **Streamlit** (User interface)

## 📂 Project Structure
```
Youtube_Assistant/
│── langchain_helper.py   # Handles transcript extraction, embedding, and query processing
│── main.py               # Streamlit UI for user interaction
│── requirements.txt      # Dependencies list
│── README.md             # Project documentation
```

## 🔧 Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/chahalbaljinder/Youtube_Assistant.git
   cd Youtube_Assistant
   ```
2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## ▶️ Usage
1. **Run the Streamlit application:**
   ```sh
   streamlit run main.py
   ```
2. **Enter a YouTube video URL** and **ask a question** about the video content.
3. **View the AI-generated response** based on the transcript.

## 📜 How It Works
1. **Extract transcript** from the provided YouTube URL.
2. **Split the transcript** into smaller chunks for better processing.
3. **Convert text into embeddings** using `SentenceTransformer`.
4. **Store and search embeddings** with FAISS for efficient retrieval.
5. **Use an LLM (Ollama)** to generate responses based on retrieved transcript content.

## 📄 Requirements
Make sure you have the following dependencies installed:
```
langchain-community
langchain
sentence-transformers
faiss-cpu
ollama
```

## 🔗 References
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)

🚀 Happy Coding! 🎥🤖

