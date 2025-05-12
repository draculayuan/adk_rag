# vector_store.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)

from .config import settings


class VectorStore:
    def __init__(self):
        aiplatform.init(
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.VERTEX_AI_LOCATION,
        )

        self._embedding_dim = settings.EMBEDDING_DIM
        #self._distance_measure_type = distance_measure_type

        # Index (create if it does not exist)
        self.index = self._get_or_create_index(
            settings.INDEX_DISPLAY_NAME
        )

        # Endpoint (create / deploy if needed)
        self.endpoint = self._get_or_create_endpoint(settings.ENDPOINT_DISPLAY_NAME)
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
        self, display_name: str
    ) -> MatchingEngineIndexEndpoint:
        """Returns an index endpoint; creates one if necessary."""
        eps = MatchingEngineIndexEndpoint.list(
            filter=f'display_name="{display_name}"'
        )

        if eps:
            return eps[0]

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

    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> None:
        """
        vectors: List of dicts each with keys → "embedding": List[float],
                                                "metadata":  Dict[str, Any]
        The ID convention below (`fileName_chunkIdx`) is only an example.
        """
        ids = [
            f"{v['metadata'].get('file_name','file')}_{v['metadata'].get('chunk_index',0)}"
            for v in vectors
        ]
        embeddings = [v["embedding"] for v in vectors]
        metadata = [v["metadata"] for v in vectors]

        # `upsert_datapoints` (Vertex AI 2.15+)
        self.index.upsert_datapoints(
            ids=ids,
            feature_vectors=embeddings,
            namespace="default",
            metadata=metadata,
        )

    def delete_vectors(self, vector_ids: List[str]) -> None:
        self.index.remove_datapoints(ids=vector_ids, namespace="default")

    # ─────────────────────────── ANN / SIMILARITY SEARCH ───────────────────────── #

    def search_vectors(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        return_full_metadata: bool = False,
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
        matches = response[0].neighbors

        results = []
        for m in matches:
            result_md = m.metadata if return_full_metadata else {
                k: m.metadata.get(k)
                for k in ("text", "file_name", "chunk_index")
                if k in m.metadata
            }
            results.append(
                {
                    "id": m.id,
                    "distance": m.distance,
                    "metadata": result_md,
                }
            )
        return results
