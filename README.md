# Cymbal Internal Knowledge Bot

A GenAI-powered internal knowledge bot that helps employees find information about company policies, onboarding documents, and technical architecture using RAG (Retrieval-Augmented Generation) architecture.

## Architecture

The system uses the following components:

1. **Vector Database**: Vertex AI Vector Search for storing and retrieving document embeddings
2. **Embedding Generation**: Vertex AI models for generating document embeddings
3. **LLM Integration**: Vertex AI Gemini for generating responses
4. **Document Processing**: Dynamic document ingestion and embedding generation
5. **Evaluation Framework**: Metrics and testing infrastructure
6. **Web Interface**: Simple UI for querying the knowledge bot

### RAG Implementation

The system implements RAG (Retrieval-Augmented Generation) using:
- Document chunking and embedding generation
- Vector similarity search using Vertex AI Vector Search
- Context-aware response generation using Vertex AI Gemini
- Source citation and reference tracking

## Setup and Installation

1. **Prerequisites**:
   - Python 3.8+
   - Google Cloud Platform account with Vertex AI API enabled
   - Google Cloud credentials configured

2. **Environment Setup**:
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configuration**:
   Create a `.env` file with the following variables:
   ```
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
   VERTEX_AI_LOCATION=us-central1
   VERTEX_AI_ENDPOINT=your-endpoint-id
   VECTOR_SEARCH_INDEX_ENDPOINT=your-index-endpoint
   VECTOR_SEARCH_INDEX_ID=your-index-id
   INDEX_DISPLAY_NAME=your-index-name
   ENDPOINT_DISPLAY_NAME=your-endpoint-name
   ENDPOINT_ID=your-endpoint-id
   STAGING_BUCKET=gs://your-bucket
   ```

## Project Structure

```
.
├── data_ingestion.py          # Document ingestion and index management
├── deploy_agent.py           # Agent deployment to Vertex AI
├── web_chatbot.py           # Web interface for the chatbot
├── src/
│   ├── agent/               # Agent implementation
│   │   ├── agent.py        # RAG agent definition
│   │   └── tools/          # Agent tools
│   ├── common/             # Shared utilities
│   │   ├── config.py       # Configuration management
│   │   ├── processor.py    # Document processing
│   │   ├── embedding_generator.py  # Embedding generation
│   │   └── vector_store.py # Vector store operations
│   └── evaluation/         # Evaluation framework
└── templates/              # Web interface templates
```

## Usage

### 1. Setting up the Vector Index

First, ingest your documents to create the vector index:

```bash
python data_ingestion.py update --path /path/to/your/documents
```

To remove specific vectors from the index:
```bash
python data_ingestion.py remove --ids vector_id1 vector_id2
```

### 2. Deploying the Agent

Deploy the RAG agent to Vertex AI Agent Engine:

```bash
python deploy_agent.py --staging-bucket gs://your-bucket
```

### 3. Starting the Web Service

Start the web interface:

```bash
python web_chatbot.py
```

The service will be available at `http://localhost:8000`

## Evaluation

The system includes a comprehensive evaluation framework using the `rag_eval.py` script, which leverages the Ragas library to assess the quality of the RAG system. The evaluation metrics include:

- **Faithfulness**: Measures if the generated answer is supported by the retrieved context
- **Answer Relevancy**: Evaluates how relevant the answer is to the question
- **Context Recall**: Measures how well the retrieved context covers the ground truth
- **Context Precision**: Assesses the precision of the retrieved context
- **Answer Correctness**: Evaluates the correctness of the answer against ground truth

### Running Evaluations

To evaluate the system, prepare a CSV file with test questions and optional ground truth answers (If you choose a metric requiring ground truth, then ground truth must be provided in the file). Then run:

```bash
python rag_eval.py --test_data path/to/test_data.csv [--metrics metric1 metric2] [--question_col column_name] [--answer_col column_name]
```

Arguments:
- `--test_data`: Path to CSV file containing test data (required)
- `--metrics`: List of metrics to evaluate (optional, defaults to all metrics)
- `--question_col`: Name of the question column in CSV (default: "question")
- `--answer_col`: Name of the ground truth answer column in CSV (default: "answer")

Example:
```bash
python rag_eval.py --test_data test_queries.csv --metrics faithfulness answer_relevancy
```

The script will output detailed evaluation results for each metric, helping you assess and improve the system's performance.

## Updating Index

The index can be updated in two ways:

1. **Adding Documents**:
   ```bash
   python data_ingestion.py update --path /path/to/new/documents
   ```

2. **Removing Documents**:
   ```bash
   python data_ingestion.py remove --ids vector_id1 vector_id2
   ```

The system supports various document formats:
- Text files (.txt)
- PDF documents (.pdf)
- Word documents (.docx)
- Markdown files (.md)
- CSV files (.csv)
- Images (.png, .jpg, .jpeg)
