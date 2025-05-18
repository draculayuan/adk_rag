# vector_store.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from google.cloud.aiplatform_v1.types import IndexDatapoint
from google.cloud import firestore

from .config import settings


class VectorStore:
    def __init__(self):
        aiplatform.init(
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.VERTEX_AI_LOCATION,
        )

        self._embedding_dim = settings.EMBEDDING_DIM
        #self._distance_measure_type = distance_measure_type
        
        # for storing metadata and text mapping
        self.db = firestore.Client()

        # Index (create if it does not exist)
        self.index = self._get_or_create_index(
            settings.INDEX_DISPLAY_NAME
        )

        # Endpoint (create / deploy if needed)
        self.endpoint = self._get_or_create_endpoint(settings.ENDPOINT_ID)
        self.deployed_index_id = (
            self._get_or_deploy_index_to_endpoint(self.endpoint)
        )

    def _get_or_create_index(self, display_name: str) -> MatchingEngineIndex:
        """Returns a MatchingEngineIndex. Creates one if it doesn’t exist."""
        matches = MatchingEngineIndex.list(
            filter=f'display_name="{display_name}"'
        )

        if matches:
            return matches[0]

        # STREAM_UPDATE means we can upsert / delete after creation.
        return MatchingEngineIndex.create_tree_ah_index(
            display_name=display_name,
            contents_delta_uri=None,  # no bulk import at creation
            dimensions=self._embedding_dim,
            approximate_neighbors_count=150, 
            #distance_measure_type=aiplatform.matching_engine.matching_engine_index_config.DistanceMeasureType.DOT_PRODUCT_DISTANCE,
            index_update_method="STREAM_UPDATE",
        )

    def _get_or_create_endpoint(
        self, endpoint_id: str
    ) -> MatchingEngineIndexEndpoint:
        """Returns an index endpoint; creates one if necessary."""
        try:
            endpoint = aiplatform.MatchingEngineIndexEndpoint(
                    index_endpoint_name=endpoint_id
                )
            return endpoint
        except:
            return MatchingEngineIndexEndpoint.create(
                display_name=display_name,
                public_endpoint_enabled=True
            )

    def _get_or_deploy_index_to_endpoint(
        self, endpoint: MatchingEngineIndexEndpoint
    ) -> str:
        """Deploys the managed index to the endpoint (if not already deployed)."""
        # Already deployed?
        for deployed in endpoint.gca_resource.deployed_indexes:
            if deployed.index == self.index.resource_name:
                return deployed.id         # ← already deployed, just reuse its ID

        # Not yet deployed – deploy now.
        deployed_index_id=f"deployed_index_{self.index.resource_name.split('/')[-1]}_v1"
        endpoint.deploy_index(
            index=self.index, deployed_index_id=deployed_index_id
        )
        return deployed_index_id

    def upsert_vectors(self, data, collection="rag") -> None:
        print("Updating index...")
        datapoints = [IndexDatapoint(
            datapoint_id=str(i), 
            feature_vector=e['embedding']) for i, e in enumerate(data)]

        # `upsert_datapoints` (Vertex AI 2.15+)
        self.index.upsert_datapoints(
            datapoints = datapoints
        )
        
        # update db
        for item in data:
            text = item['text']
            metadata = item['metadata']
            idx = metadata['chunk_index']
            file_name = metadata['file_name']
            source = metadata['source']
            doc_ref = self.db.collection(collection).document(str(idx))
            doc_ref.set({
                "file_path": source,
                "file_name": file_name,
                "text": item["text"]
            })

    def delete_vectors(self, vector_ids: List[str], collection="rag") -> None:
        self.index.remove_datapoints(datapoint_ids=vector_ids)
        for idx in vector_ids:
            self.db.collection(collection).document(idx).delete()

    # ─────────────────────────── ANN / SIMILARITY SEARCH ───────────────────────── #

    def search_vectors(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        collection="rag",
    ) -> List[Dict[str, Any]]:
        """
        Runs an ANN lookup against the deployed index endpoint.
        """
        response = self.endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_embedding],
            num_neighbors=top_k,
        )

        # Only one query, so response is a single MatchResponse.
        try:
            retrieved_results = []
            for response_ in response[0]:
                r = response_.id
                r = self.db.collection(collection).document(r).get().to_dict()
                retrieved_results.append(
                    {
                        'text': r['text'],
                        'file_name': r['file_name'],
                        'file_path': r['file_path']
                    }
                )
        except:
            retrieved_results = [{
                "text": "", 
                'file_name': "",
                'file_path': ""
            }]
        return retrieved_results
