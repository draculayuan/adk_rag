import json
import os
import argparse
from evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system without ground truth")
    parser.add_argument("--queries", "-q", help="Path to JSON file with test queries")
    parser.add_argument("--num-samples", "-n", type=int, default=5, 
                        help="Number of sample queries to generate if no query file provided")
    args = parser.parse_args()
    
    # Get test queries
    if args.queries and os.path.exists(args.queries):
        # Load queries from file
        with open(args.queries, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "queries" in data:
                test_queries = data["queries"]
            elif isinstance(data, list):
                test_queries = data
            else:
                raise ValueError("Query file should contain a list of queries or a dict with a 'queries' key")
    else:
        # Use sample queries if no file provided
        test_queries = generate_sample_queries(args.num_samples)
    
    print(f"Running evaluation on {len(test_queries)} queries...")
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Run evaluation
    results = evaluator.evaluate(test_queries)
    
    # Print metrics
    print("\nEvaluation Metrics (No Ground Truth):")
    print("-" * 60)
    for metric, value in results["metrics"].items():
        if "score" in metric:
            print(f"{metric}: {value:.4f} (higher is better)")
        else:
            print(f"{metric}: {value:.2f}")
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 60)
    for i, result in enumerate(results["results"]):
        print(f"\n[Query {i+1}] {result['query']}")
        print(f"\nRetrieval score: {result['retrieval_score']:.4f}")
        print(f"Overall quality score: {result['overall_score']:.4f}")
        print(f"Retrieved chunks: {result['retrieved_chunks_count']}")
        print(f"\nResponse: {result['response']}")
        print("\n" + "-" * 30)

def generate_sample_queries(num_samples=5):
    """Generate sample queries for evaluation if no query file is provided."""
    sample_queries = [
        "What is the company's vacation policy?",
        "How do I set up my development environment?",
        "What is the process for submitting a pull request?",
        "What are the core values of the company?",
        "How do I request time off?",
        "What's the maximum reimbursement for professional development?",
        "Who should I contact for IT support?",
        "When is the next company-wide meeting?",
        "What is our policy on remote work?",
        "How do I access the company's knowledge base?"
    ]
    
    # Return requested number of samples (or all if num_samples > available)
    return sample_queries[:min(num_samples, len(sample_queries))]

if __name__ == "__main__":
    main() 