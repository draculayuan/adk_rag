from typing import List, Dict, Any
import json
import os
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from vertexai.generative_models import GenerativeModel

from ..common.embedding_generator import EmbeddingGenerator
from ..common.vector_store import VectorStore
from ..common.config import settings

class Evaluator:
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        # Use Gemini model for evaluation
        self.model = GenerativeModel(settings.LLM_MODEL)
        
    def evaluate(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        Evaluate the RAG system using metrics that don't require ground truth.
        
        Args:
            test_queries: List of query strings to evaluate
            
        Returns:
            Dictionary containing evaluation results and metrics
        """
        results = []
        
        for query in test_queries:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(query)
            
            # Search for relevant chunks
            retrieved_chunks = self.vector_store.search_vectors(query_embedding, top_k=5)
            
            # Extract text from retrieved chunks
            retrieved_texts = [chunk['metadata'].get('text', '') for chunk in retrieved_chunks]
            
            # Generate response using retrieved chunks
            response = self._generate_response(query, retrieved_texts)
            
            # Evaluate retrieval quality
            retrieval_score = self._evaluate_retrieval_quality(query, retrieved_texts)
            
            # Evaluate overall quality
            overall_score = self._evaluate_overall_quality(query, response, retrieved_texts)
            
            results.append({
                "query": query,
                "response": response,
                "retrieval_score": retrieval_score,
                "overall_score": overall_score,
                "retrieved_chunks_count": len(retrieved_texts),
                "retrieved_chunks": retrieved_texts[:2]  # Only include first two chunks to save space
            })
        
        # Calculate aggregated metrics
        metrics = self._calculate_metrics(results)
        
        # Save results
        self._save_results(results, metrics)
        
        return {
            "results": results,
            "metrics": metrics
        }
    
    def _generate_response(self, query: str, context_texts: List[str]) -> str:
        """Generate a response to the query using the retrieved context."""
        context = "\n\n".join(context_texts)
        prompt = f"""
        Based on the following information, please answer the query:
        
        QUERY: {query}
        
        CONTEXT:
        {context}
        
        ANSWER:
        """
        
        response = self.model.generate_content(prompt)
        return response.text.strip()
    
    def _evaluate_retrieval_quality(self, query: str, retrieved_texts: List[str]) -> float:
        """
        Evaluate retrieval quality using semantic similarity between query and retrieved chunks.
        
        This measures how relevant the retrieved chunks are to the query.
        Higher score means better retrieval quality.
        """
        if not retrieved_texts:
            return 0.0
            
        # Get embeddings for query and retrieved texts
        query_embedding = self.embedding_generator.generate_single_embedding(query)
        
        chunk_embeddings = []
        for text in retrieved_texts:
            if text.strip():  # Skip empty texts
                chunk_embeddings.append(self.embedding_generator.generate_single_embedding(text))
        
        if not chunk_embeddings:
            return 0.0
            
        # Calculate cosine similarities between query and each chunk
        similarities = []
        for emb in chunk_embeddings:
            # Reshape embeddings for sklearn cosine_similarity
            query_emb_reshaped = np.array(query_embedding).reshape(1, -1)
            chunk_emb_reshaped = np.array(emb).reshape(1, -1)
            sim = cosine_similarity(query_emb_reshaped, chunk_emb_reshaped)[0][0]
            similarities.append(sim)
        
        # Use average of top 3 similarities (or all if less than 3)
        similarities.sort(reverse=True)
        top_similarities = similarities[:min(3, len(similarities))]
        avg_similarity = sum(top_similarities) / len(top_similarities)
        
        return float(avg_similarity)
    
    def _evaluate_overall_quality(self, query: str, response: str, retrieved_texts: List[str]) -> float:
        """
        Evaluate overall quality using a LLM-based coherence score.
        
        This measures how well the response uses the retrieved information
        and how coherent and factual it is.
        """
        if not response or not retrieved_texts:
            return 0.0
            
        context = "\n\n".join(retrieved_texts)
        
        # LLM-based evaluation prompt
        evaluation_prompt = f"""
        You are evaluating the quality of a response to a query in a retrieval-augmented generation system.
        Rate the response on a scale from 0.0 to 1.0 based on:
        
        1. Coherence: Is the response well-structured and logically connected?
        2. Informativeness: Does the response provide substantive information?
        3. Relevance: Does the response directly address the query?
        4. Faithfulness: Does the response stick to information in the retrieved context?
        
        QUERY: {query}
        
        RETRIEVED CONTEXT:
        {context}
        
        RESPONSE:
        {response}
        
        Evaluate each criterion and then provide a final score between 0.0 and 1.0.
        Your output should be ONLY the numeric score, with no explanations or other text.
        """
        
        try:
            evaluation = self.model.generate_content(evaluation_prompt, temperature=0.0)
            score_text = evaluation.text.strip()
            
            # Extract numeric score from response
            try:
                score = float(score_text)
                # Ensure score is within valid range
                score = max(0.0, min(1.0, score))
                return score
            except ValueError:
                # If parsing fails, return a moderate score
                return 0.5
                
        except Exception as e:
            print(f"Error in overall quality evaluation: {e}")
            return 0.5
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregated evaluation metrics."""
        if not results:
            return {
                "avg_retrieval_score": 0.0,
                "avg_overall_score": 0.0,
                "avg_retrieved_chunks": 0.0
            }
            
        retrieval_scores = [r["retrieval_score"] for r in results]
        overall_scores = [r["overall_score"] for r in results]
        chunk_counts = [r["retrieved_chunks_count"] for r in results]
        
        metrics = {
            "avg_retrieval_score": sum(retrieval_scores) / len(retrieval_scores),
            "avg_overall_score": sum(overall_scores) / len(overall_scores),
            "avg_retrieved_chunks": sum(chunk_counts) / len(chunk_counts)
        }
        
        return metrics
    
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