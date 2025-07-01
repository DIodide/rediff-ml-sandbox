"""
Utility functions for Pinecone Vector Database operations.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    raise ImportError(
        "Pinecone client not installed. Install with: pip install pinecone-client"
    )

logger = logging.getLogger(__name__)


@dataclass
class PineconeConfig:
    """Configuration for Pinecone connection."""

    api_key: str
    environment: Optional[str] = None
    index_name: str = "ml-sandbox-index"
    dimension: int = 1536  # OpenAI ada-002 dimension
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"


class PineconeManager:
    """Manager class for Pinecone operations."""

    def __init__(self, config: PineconeConfig):
        """
        Initialize Pinecone manager.

        Args:
            config: Pinecone configuration
        """
        self.config = config
        self.pc = None
        self.index = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Pinecone client."""
        try:
            self.pc = Pinecone(api_key=self.config.api_key)
            logger.info("Pinecone client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise

    def create_index(
        self,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        metric: Optional[str] = None,
        spec_type: str = "serverless",
    ) -> bool:
        """
        Create a new Pinecone index.

        Args:
            index_name: Name of the index
            dimension: Vector dimension
            metric: Distance metric
            spec_type: Index spec type ('serverless' or 'pod')

        Returns:
            Success status
        """
        index_name = index_name or self.config.index_name
        dimension = dimension or self.config.dimension
        metric = metric or self.config.metric

        try:
            # Check if index already exists
            if index_name in self.pc.list_indexes().names():
                logger.info(f"Index '{index_name}' already exists")
                return True

            # Create spec based on type
            if spec_type == "serverless":
                spec = ServerlessSpec(
                    cloud=self.config.cloud, region=self.config.region
                )
            else:
                # For pod-based indexes
                spec = {"pod": {"environment": self.config.environment}}

            # Create index
            self.pc.create_index(
                name=index_name, dimension=dimension, metric=metric, spec=spec
            )

            # Wait for index to be ready
            while not self.pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

            logger.info(f"Index '{index_name}' created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False

    def connect_to_index(self, index_name: Optional[str] = None):
        """
        Connect to an existing index.

        Args:
            index_name: Name of the index to connect to
        """
        index_name = index_name or self.config.index_name

        try:
            self.index = self.pc.Index(index_name)
            logger.info(f"Connected to index '{index_name}'")
        except Exception as e:
            logger.error(f"Failed to connect to index: {e}")
            raise

    def upsert_vectors(
        self,
        vectors: List[Tuple[str, List[float], Dict[str, Any]]] = None,
        df: pd.DataFrame = None,
        id_column: str = "id",
        vector_column: str = "vector",
        metadata_columns: List[str] = None,
        batch_size: int = 100,
    ) -> bool:
        """
        Upsert vectors to the index.

        Args:
            vectors: List of (id, vector, metadata) tuples
            df: DataFrame with vectors and metadata
            id_column: Column name for IDs in DataFrame
            vector_column: Column name for vectors in DataFrame
            metadata_columns: List of metadata column names
            batch_size: Batch size for upserts

        Returns:
            Success status
        """
        if self.index is None:
            raise ValueError("No index connected. Call connect_to_index() first.")

        try:
            # Prepare vectors from DataFrame if provided
            if df is not None:
                vectors = []
                metadata_columns = metadata_columns or []

                for _, row in df.iterrows():
                    vector_id = str(row[id_column])
                    vector = row[vector_column]

                    # Convert numpy array to list if needed
                    if isinstance(vector, np.ndarray):
                        vector = vector.tolist()

                    # Prepare metadata
                    metadata = {col: row[col] for col in metadata_columns if col in row}

                    vectors.append((vector_id, vector, metadata))

            # Upsert in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(
                    f"Upserted batch {i // batch_size + 1}/{(len(vectors) + batch_size - 1) // batch_size}"
                )

            logger.info(f"Successfully upserted {len(vectors)} vectors")
            return True

        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            return False

    def query_vectors(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> Dict[str, Any]:
        """
        Query the index for similar vectors.

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filter_dict: Metadata filter
            include_metadata: Include metadata in response
            include_values: Include vector values in response

        Returns:
            Query results
        """
        if self.index is None:
            raise ValueError("No index connected. Call connect_to_index() first.")

        try:
            # Convert numpy array to list if needed
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()

            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=include_metadata,
                include_values=include_values,
            )

            return response

        except Exception as e:
            logger.error(f"Failed to query vectors: {e}")
            raise

    def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs.

        Args:
            ids: List of vector IDs to delete

        Returns:
            Success status
        """
        if self.index is None:
            raise ValueError("No index connected. Call connect_to_index() first.")

        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors")
            return True

        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Index statistics
        """
        if self.index is None:
            raise ValueError("No index connected. Call connect_to_index() first.")

        try:
            stats = self.index.describe_index_stats()
            return stats

        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}

    def list_indexes(self) -> List[str]:
        """
        List all available indexes.

        Returns:
            List of index names
        """
        try:
            return self.pc.list_indexes().names()
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            return []

    def delete_index(self, index_name: Optional[str] = None) -> bool:
        """
        Delete an index.

        Args:
            index_name: Name of index to delete

        Returns:
            Success status
        """
        index_name = index_name or self.config.index_name

        try:
            self.pc.delete_index(index_name)
            logger.info(f"Deleted index '{index_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            return False


def create_pinecone_manager_from_env() -> PineconeManager:
    """
    Create PineconeManager from environment variables.

    Environment variables:
    - PINECONE_API_KEY: Pinecone API key
    - PINECONE_ENVIRONMENT: Pinecone environment (optional)
    - PINECONE_INDEX_NAME: Default index name

    Returns:
        PineconeManager instance
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")

    config = PineconeConfig(
        api_key=api_key,
        environment=os.getenv("PINECONE_ENVIRONMENT"),
        index_name=os.getenv("PINECONE_INDEX_NAME", "ml-sandbox-index"),
    )

    return PineconeManager(config)


def vectors_to_pinecone_format(
    df: pd.DataFrame,
    id_column: str,
    vector_column: str,
    metadata_columns: List[str] = None,
) -> List[Tuple[str, List[float], Dict[str, Any]]]:
    """
    Convert DataFrame to Pinecone format.

    Args:
        df: Input DataFrame
        id_column: Column name for IDs
        vector_column: Column name for vectors
        metadata_columns: List of metadata column names

    Returns:
        List of (id, vector, metadata) tuples
    """
    vectors = []
    metadata_columns = metadata_columns or []

    for _, row in df.iterrows():
        vector_id = str(row[id_column])
        vector = row[vector_column]

        # Convert numpy array to list if needed
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        # Prepare metadata
        metadata = {col: row[col] for col in metadata_columns if col in row}

        vectors.append((vector_id, vector, metadata))

    return vectors
