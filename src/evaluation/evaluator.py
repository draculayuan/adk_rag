from typing import List, Dict, Any
import json
import os
from datetime import datetime
from ..embedding.embedding_generator import EmbeddingGenerator
from ..vector_store.vector_store import VectorStore
from ..llm.llm_engine import LLMEngine

class Evaluator:
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.llm_engine = LLMEngine()
        
    def evaluate(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the knowledge bot against test cases."""
        results = []
        
        for test_case in test_cases:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(test_case["query"])
            
            # Search for relevant chunks
            relevant_chunks = self.vector_store.search_vectors(query_embedding)
            
            # Generate response
            response = self.llm_engine.generate_response(test_case["query"], relevant_chunks)
            
            # Evaluate response
            evaluation = self._evaluate_response(
                response["answer"],
                test_case["expected_answer"],
                relevant_chunks
            )
            
            results.append({
                "query": test_case["query"],
                "response": response["answer"],
                "expected_answer": test_case["expected_answer"],
                "evaluation": evaluation,
                "sources": response["sources"]
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        # Save results
        self._save_results(results, metrics)
        
        return {
            "results": results,
            "metrics": metrics
        }
    
    def _evaluate_response(self, response: str, expected: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a single response."""
        # This is a simple evaluation - in a real system, you might want to use
        # more sophisticated metrics like ROUGE, BLEU, or semantic similarity
        return {
            "exact_match": response.strip().lower() == expected.strip().lower(),
            "context_relevance": len(context) > 0,
            "source_count": len(context)
        }
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        total = len(results)
        exact_matches = sum(1 for r in results if r["evaluation"]["exact_match"])
        relevant_context = sum(1 for r in results if r["evaluation"]["context_relevance"])
        avg_sources = sum(r["evaluation"]["source_count"] for r in results) / total
        
        return {
            "exact_match_rate": exact_matches / total,
            "context_relevance_rate": relevant_context / total,
            "average_sources_per_query": avg_sources
        }
    
    def _save_results(self, results: List[Dict[str, Any]], metrics: Dict[str, float]):
        """Save evaluation results to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        output = {
            "timestamp": timestamp,
            "results": results,
            "metrics": metrics
        }
        
        with open(f"{output_dir}/evaluation_{timestamp}.json", "w") as f:
            json.dump(output, f, indent=2) 