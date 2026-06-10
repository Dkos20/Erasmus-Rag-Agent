# Erasmus+ Local RAG Chatbot

This project implements a fully containerized, local Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about the Erasmus+ mobility program in both Croatian and English. 

The system operates entirely on CPU and is composed of three interconnected services: a Large Language Model runner, an API gateway proxy, and a web-based chat interface.

---

## Architecture & Technology Stack

The application is containerized using Docker Compose and contains the following stack:
1. **Ollama**: Serves the local LLM (`llama3:8b-instruct-q4_K_M`) using CPU execution.
2. **LiteLLM**: Acts as an OpenAI-compatible API gateway proxy to route requests to the Ollama container.
3. **Streamlit (Frontend)**: Provides a web interface for users to select their UI language, automatically load the vector database, and interact with the chatbot in real time.
4. **LangChain & FAISS (RAG Backend)**: Processes documents from the host machine, splits them, embeds them using a lightweight multilingual model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`), and performs similarity searches.

---

## System Evaluation Suite

The application includes an automated evaluation pipeline that scores the chatbot’s performance against a test suite of 5 benchmark questions based on actual document content. 

The evaluation calculates three custom metrics from scratch:
- **Semantic Vector Cosine Similarity**: Measures the semantic similarity between the generated response and the ground-truth reference answer using the `SentenceTransformer` embeddings.
- **Hit Rate @ 3**: Checks whether the correct source file containing the answer was retrieved in the top 3 FAISS context chunks.
- **Key Entity Accuracy**: Checks the percentage of specific expected keywords/numbers/phrases present in the generated chatbot response.

### Automated Score & Feedback
Based on the metrics above, the system calculates a weighted overall quality score from **1.0 to 5.0** using the formula:
$$\text{Score} = 1.0 + 4.0 \times (0.5 \times \text{Semantic Cosine Similarity} + 0.3 \times \text{Key Entity Accuracy} + 0.2 \times \text{Hit Rate @ 3})$$
The system also generates an automated, localized assessment comment (in Croatian or English depending on the question language) detailing the quality of the response.

---

## Project Structure

```text
├── data/                    # Host-mounted folder containing source documents (.txt, .pdf)
├── vectorstore/             # Host-mounted folder where the FAISS index and evaluation results are persisted
├── app.py                   # Streamlit web application, chat interface, and evaluation UI
├── ingest.py                # LangChain ingestion script for document preprocessing
├── evaluate.py              # Automated RAG evaluation and scoring script
├── evaluation_set.json      # Evaluation benchmark dataset containing questions and reference answers
├── Dockerfile               # Build configuration for the Streamlit application container
├── requirements.txt         # Python library dependencies (optimized for CPU execution)
├── litellm-config.yaml      # LiteLLM routing configuration mapping 'llama3' to Ollama
├── docker-compose.yml       # Docker orchestration file for the three services
└── README.md                # Project documentation
```

---

## How to Run the Project

Ensure you have Docker and Docker Compose installed on your host system.

### 1. Build and Start the Containers
Start the services in the background:
```bash
docker compose up -d
```
On the initial startup, the container will automatically run `ingest.py` to index the documents in `./data` and store them in `./vectorstore/db_faiss`.

### 2. Pull the LLM Model
Execute the following command to download the Llama3 model inside the running Ollama container:
```bash
docker exec -it ollama ollama pull llama3:8b-instruct-q4_K_M
```

### 3. Open the Web Application
Once the containers are running and the model is downloaded, access the Streamlit UI by opening your browser at:
```
http://localhost:8501
```

### 4. Indexing & Chatting
- **Auto-Ingestion**: The system automatically scans the `./data` directory at startup and loads the FAISS index. You do not need to click any buttons to start.
- **Manual Reindexing**: If you add or modify documents in the `./data` folder, click the **Start Document Indexing** (or **Pokreni vektorizaciju dokumenata**) button in the sidebar to rebuild the index.
- **Evaluation**: Switch to the **System Evaluation** tab and click **Run System Evaluation** (or **Pokreni evaluaciju sustava**) to run the automated test suite and view the metrics table.
