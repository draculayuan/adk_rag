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

The RAG pipeline consists of:
- Document ingestion and chunking
- Embedding generation using Vertex AI
- Vector storage in Vertex AI Vector Search
- Context retrieval and response generation
- Source traceability for answers

## Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd cymbal-knowledge-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Google Cloud credentials and configuration
```

4. Run the application:
```bash
python src/main.py
```

## Project Structure

```
cymbal-knowledge-bot/
├── src/
│   ├── main.py                 # Application entry point
│   ├── config.py              # Configuration management
│   ├── document_processor/    # Document processing modules
│   ├── embedding/            # Embedding generation
│   ├── vector_store/        # Vector database operations
│   ├── llm/                # LLM integration
│   ├── evaluation/        # Evaluation framework
│   └── web/              # Web interface
├── tests/               # Test suite
├── data/               # Sample documents and test data
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Usage

1. Start the application:
```bash
python src/main.py
```

2. Access the web interface at `http://localhost:8000`

3. Enter your question in natural language

4. View the response along with source documents used

## Evaluation

The evaluation framework can be run using:
```bash
python src/evaluation/run_evaluation.py
```

This will:
- Test the system against a set of sample questions
- Generate accuracy metrics
- Provide insights on response quality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

[License Information] 