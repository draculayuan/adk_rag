import json
import os
from evaluator import Evaluator

def main():
    # Load test cases
    test_cases_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    with open(test_cases_path, "r") as f:
        test_data = json.load(f)
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Run evaluation
    results = evaluator.evaluate(test_data["test_cases"])
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print("-" * 50)
    for metric, value in results["metrics"].items():
        print(f"{metric}: {value:.2%}")
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 50)
    for result in results["results"]:
        print(f"\nQuery: {result['query']}")
        print(f"Expected: {result['expected_answer']}")
        print(f"Response: {result['response']}")
        print(f"Evaluation: {result['evaluation']}")
        print(f"Sources: {result['sources']}")

if __name__ == "__main__":
    main() 