from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
)
import pandas as pd
import argparse
import sys
from typing import List, Dict, Any, Optional
import vertexai
from vertexai import agent_engines

# Initialize the agent engine
agent_engine = vertexai.agent_engines.get(
    "projects/163097687798/locations/us-central1/reasoningEngines/8537074470983041024"
)
session = agent_engine.create_session(user_id="test_user")

# Available metrics mapping
AVAILABLE_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_recall": context_recall,
    "context_precision": context_precision,
    "answer_correctness": answer_correctness,
}

# Metrics that require ground truth
GROUND_TRUTH_METRICS = {
    "context_recall",
    "answer_correctness"
}

def call_agent(query: str) -> tuple[str, list[str]]:
    """
    Call the agent with a query and return the response and contexts.
    
    Args:
        query (str): The query to send to the agent
        
    Returns:
        tuple[str, list[str]]: The response text and list of contexts
    """
    response_text = ""
    contexts = []
    for event in agent_engine.stream_query(
        user_id="test_user", session_id=session["id"], message=query
    ):
        text = event["content"]["parts"][0].get("text", None)
        function_response = event["content"]["parts"][0].get("function_response", None)
        role = event["content"]["role"]
        if (text is not None) and (role == "model"):
            response_text += text
        elif function_response is not None:
            contexts.extend(
                [r["text"] for r in function_response["response"]["result"]]
            )
    return response_text, contexts

def run_rag(q: str) -> tuple[str, list[str]]:
    """
    Run RAG with retry logic for rate limiting.
    
    Args:
        q (str): The query to process
        
    Returns:
        tuple[str, list[str]]: The answer and contexts
    """
    while True:  # handle rate limit
        try:
            answer, contexts = call_agent(q)
            break
        except Exception as e:
            print(f"Error occurred, retrying: {str(e)}")
            continue
    return answer, contexts

def load_test_data(file_path: str, question_col: str = "question", answer_col: str = "answer", require_answer: bool = False) -> List[Dict[str, str]]:
    """
    Load test data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        question_col (str): Name of the column containing questions
        answer_col (str): Name of the column containing ground truth answers
        require_answer (bool): Whether the answer column is required
        
    Returns:
        List[Dict[str, str]]: List of test samples
    """
    try:
        test_data = pd.read_csv(file_path)
        
        # Validate question column exists
        if question_col not in test_data.columns:
            raise ValueError(f"Missing required column '{question_col}' in CSV. Available columns: {list(test_data.columns)}")
        
        # Validate answer column if required
        if require_answer and answer_col not in test_data.columns:
            raise ValueError(f"Selected metrics require ground truth answers. Missing column '{answer_col}' in CSV. Available columns: {list(test_data.columns)}")
        
        # Create test samples
        samples = []
        for _, row in test_data.iterrows():
            sample = {"question": row[question_col]}
            if answer_col in test_data.columns:
                sample["ground_truth"] = row[answer_col]
            samples.append(sample)
        
        return samples
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        sys.exit(1)

def get_selected_metrics(metric_names: Optional[List[str]] = None) -> List[Any]:
    """
    Get the selected metrics based on user input.
    
    Args:
        metric_names (Optional[List[str]]): List of metric names to use
        
    Returns:
        List[Any]: List of metric functions
    """
    if metric_names is None:
        return list(AVAILABLE_METRICS.values())
    
    selected_metrics = []
    for name in metric_names:
        if name in AVAILABLE_METRICS:
            selected_metrics.append(AVAILABLE_METRICS[name])
        else:
            print(f"Warning: Metric '{name}' not found. Available metrics: {list(AVAILABLE_METRICS.keys())}")
    
    if not selected_metrics:
        print("No valid metrics selected. Using all available metrics.")
        return list(AVAILABLE_METRICS.values())
    
    return selected_metrics

def requires_ground_truth(metrics: List[Any]) -> bool:
    """
    Check if any of the selected metrics require ground truth.
    
    Args:
        metrics (List[Any]): List of metric functions
        
    Returns:
        bool: True if any metric requires ground truth
    """
    metric_names = [name for name, func in AVAILABLE_METRICS.items() if func in metrics]
    return any(name in GROUND_TRUTH_METRICS for name in metric_names)

def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Script")
    parser.add_argument("--test_data", type=str, required=True,
                      help="Path to the CSV file containing test data")
    parser.add_argument("--metrics", type=str, nargs="+",
                      help="List of metrics to use (default: all available metrics)",
                      choices=list(AVAILABLE_METRICS.keys()))
    parser.add_argument("--question_col", type=str, default="question",
                      help="Name of the column containing questions in the CSV (default: 'question')")
    parser.add_argument("--answer_col", type=str, default="answer",
                      help="Name of the column containing ground truth answers in the CSV (default: 'answer')")
    
    args = parser.parse_args()
    
    # Get selected metrics
    selected_metrics = get_selected_metrics(args.metrics)
    
    # Check if ground truth is required
    needs_ground_truth = requires_ground_truth(selected_metrics)
    
    # Load test data
    TEST_SET = load_test_data(args.test_data, args.question_col, args.answer_col, needs_ground_truth)
    
    # Process each test sample
    records = []
    for sample in TEST_SET:
        ans, ctxs = run_rag(sample["question"])
        record = {
            "question": sample["question"],
            "answer": ans,
            "contexts": ctxs,
        }
        if "ground_truth" in sample:
            record["ground_truth"] = sample["ground_truth"]
        records.append(record)

    eval_ds = Dataset.from_list(records)

    # Initialize LLM and embeddings
    chat_llm = ChatVertexAI(
        model_name="gemini-2.5-pro-preview-05-06",
        project="yuan-449301",
        location="us-central1",
        temperature=0,
    )
    llm = LangchainLLMWrapper(chat_llm)

    lc_embed = VertexAIEmbeddings(
        model_name="text-embedding-005"
    )
    embedder = LangchainEmbeddingsWrapper(lc_embed)

    # Run evaluation
    results = evaluate(
        eval_ds,
        metrics=selected_metrics,
        embeddings=embedder,
        llm=llm,
    )

    print("\nEvaluation Results:")
    print(results)

if __name__ == "__main__":
    main()
