"""
Utility functions for Supabase/PostgreSQL database operations.
"""

import os
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import logging

try:
    from supabase import create_client, Client
    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker, declarative_base
    import asyncpg
    import psycopg2
    from psycopg2.extras import Json
except ImportError as e:
    raise ImportError(
        f"Required packages not installed: {e}. Install with the provided requirements.txt"
    )

logger = logging.getLogger(__name__)


@dataclass
class SupabaseConfig:
    """Configuration for Supabase connection."""

    url: str
    key: str
    db_url: Optional[str] = None  # Direct PostgreSQL connection
    schema: str = "public"


class SupabaseManager:
    """Manager class for Supabase operations."""

    def __init__(self, config: SupabaseConfig):
        """
        Initialize Supabase manager.

        Args:
            config: Supabase configuration
        """
        self.config = config
        self.client: Optional[Client] = None
        self.engine = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Supabase client."""
        try:
            self.client = create_client(self.config.url, self.config.key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise

    def setup_sqlalchemy_engine(self, async_engine: bool = False):
        """
        Set up SQLAlchemy engine for direct database access.

        Args:
            async_engine: Whether to create an async engine
        """
        if not self.config.db_url:
            logger.warning("No database URL provided for SQLAlchemy engine")
            return

        try:
            if async_engine:
                self.engine = create_async_engine(
                    self.config.db_url.replace(
                        "postgresql://", "postgresql+asyncpg://"
                    ),
                    echo=False,
                )
            else:
                self.engine = sa.create_engine(self.config.db_url, echo=False)

            logger.info("SQLAlchemy engine created successfully")
        except Exception as e:
            logger.error(f"Failed to create SQLAlchemy engine: {e}")
            raise

    def insert_data(
        self,
        table_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        upsert: bool = False,
        conflict_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Insert data into a Supabase table.

        Args:
            table_name: Name of the table
            data: Data to insert
            upsert: Whether to perform upsert on conflict
            conflict_columns: Columns to check for conflicts in upsert

        Returns:
            Response from Supabase
        """
        try:
            if upsert:
                response = (
                    self.client.table(table_name)
                    .upsert(
                        data,
                        on_conflict=",".join(conflict_columns)
                        if conflict_columns
                        else None,
                    )
                    .execute()
                )
            else:
                response = self.client.table(table_name).insert(data).execute()

            logger.info(f"Successfully inserted data into {table_name}")
            return response.data

        except Exception as e:
            logger.error(f"Failed to insert data into {table_name}: {e}")
            raise

    def query_data(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query data from a Supabase table.

        Args:
            table_name: Name of the table
            columns: Columns to select
            filters: Filters to apply
            order_by: Column to order by
            limit: Maximum number of records
            offset: Number of records to skip

        Returns:
            Query results
        """
        try:
            query = self.client.table(table_name)

            # Select columns
            if columns:
                query = query.select(",".join(columns))
            else:
                query = query.select("*")

            # Apply filters
            if filters:
                for column, value in filters.items():
                    if isinstance(value, dict):
                        # Handle complex filters like {"gte": 10} or {"in": [1,2,3]}
                        for op, val in value.items():
                            if op == "eq":
                                query = query.eq(column, val)
                            elif op == "neq":
                                query = query.neq(column, val)
                            elif op == "gt":
                                query = query.gt(column, val)
                            elif op == "gte":
                                query = query.gte(column, val)
                            elif op == "lt":
                                query = query.lt(column, val)
                            elif op == "lte":
                                query = query.lte(column, val)
                            elif op == "like":
                                query = query.like(column, val)
                            elif op == "ilike":
                                query = query.ilike(column, val)
                            elif op == "in":
                                query = query.in_(column, val)
                            elif op == "is":
                                query = query.is_(column, val)
                    else:
                        # Simple equality filter
                        query = query.eq(column, value)

            # Order by
            if order_by:
                if order_by.startswith("-"):
                    query = query.order(order_by[1:], desc=True)
                else:
                    query = query.order(order_by)

            # Limit and offset
            if limit:
                query = query.limit(limit)
            if offset:
                query = query.offset(offset)

            response = query.execute()
            return response.data

        except Exception as e:
            logger.error(f"Failed to query data from {table_name}: {e}")
            raise

    def update_data(
        self, table_name: str, data: Dict[str, Any], filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update data in a Supabase table.

        Args:
            table_name: Name of the table
            data: Data to update
            filters: Filters to identify records to update

        Returns:
            Response from Supabase
        """
        try:
            query = self.client.table(table_name).update(data)

            # Apply filters
            for column, value in filters.items():
                query = query.eq(column, value)

            response = query.execute()
            logger.info(f"Successfully updated data in {table_name}")
            return response.data

        except Exception as e:
            logger.error(f"Failed to update data in {table_name}: {e}")
            raise

    def delete_data(self, table_name: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete data from a Supabase table.

        Args:
            table_name: Name of the table
            filters: Filters to identify records to delete

        Returns:
            Response from Supabase
        """
        try:
            query = self.client.table(table_name).delete()

            # Apply filters
            for column, value in filters.items():
                query = query.eq(column, value)

            response = query.execute()
            logger.info(f"Successfully deleted data from {table_name}")
            return response.data

        except Exception as e:
            logger.error(f"Failed to delete data from {table_name}: {e}")
            raise

    def execute_sql(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute raw SQL query using Supabase RPC.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            Query results
        """
        try:
            # Note: This requires creating a stored function in Supabase
            # For direct SQL execution, use the SQLAlchemy methods instead
            response = self.client.rpc(
                "execute_sql", {"query": query, "params": params or {}}
            ).execute()
            return response.data

        except Exception as e:
            logger.error(f"Failed to execute SQL query: {e}")
            raise

    def bulk_insert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        batch_size: int = 1000,
        upsert: bool = False,
        conflict_columns: Optional[List[str]] = None,
    ) -> bool:
        """
        Bulk insert DataFrame into Supabase table.

        Args:
            df: DataFrame to insert
            table_name: Name of the table
            batch_size: Size of each batch
            upsert: Whether to perform upsert
            conflict_columns: Columns to check for conflicts

        Returns:
            Success status
        """
        try:
            # Convert DataFrame to list of dicts
            records = df.to_dict("records")

            # Convert numpy types to native Python types
            for record in records:
                for key, value in record.items():
                    if isinstance(value, np.integer):
                        record[key] = int(value)
                    elif isinstance(value, np.floating):
                        record[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        record[key] = value.tolist()
                    elif pd.isna(value):
                        record[key] = None

            # Insert in batches
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                self.insert_data(
                    table_name, batch, upsert=upsert, conflict_columns=conflict_columns
                )
                logger.info(
                    f"Inserted batch {i // batch_size + 1}/{(len(records) + batch_size - 1) // batch_size}"
                )

            logger.info(
                f"Successfully bulk inserted {len(records)} records into {table_name}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to bulk insert DataFrame: {e}")
            return False

    def query_to_dataframe(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Query data and return as DataFrame.

        Args:
            table_name: Name of the table
            columns: Columns to select
            filters: Filters to apply
            order_by: Column to order by
            limit: Maximum number of records

        Returns:
            DataFrame with query results
        """
        try:
            data = self.query_data(table_name, columns, filters, order_by, limit)
            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Failed to query to DataFrame: {e}")
            raise

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Get table schema information.

        Args:
            table_name: Name of the table

        Returns:
            Table schema information
        """
        try:
            # This would typically require a custom RPC function or direct PostgreSQL access
            # For now, we'll use a simple approach
            sample_data = self.query_data(table_name, limit=1)
            if sample_data:
                return {col: type(val).__name__ for col, val in sample_data[0].items()}
            else:
                return {}

        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            return {}


class PostgreSQLManager:
    """Manager for direct PostgreSQL operations (useful for complex queries)."""

    def __init__(self, connection_string: str):
        """
        Initialize PostgreSQL manager.

        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string
        self.engine = sa.create_engine(connection_string)

    def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Query results as DataFrame
        """
        try:
            return pd.read_sql(query, self.engine, params=params)
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise

    def insert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
        index: bool = False,
        method: Optional[str] = None,
    ) -> bool:
        """
        Insert DataFrame into PostgreSQL table.

        Args:
            df: DataFrame to insert
            table_name: Name of the table
            if_exists: What to do if table exists ('fail', 'replace', 'append')
            index: Whether to write DataFrame index
            method: Method to use for insertion

        Returns:
            Success status
        """
        try:
            df.to_sql(
                table_name, self.engine, if_exists=if_exists, index=index, method=method
            )
            logger.info(f"Successfully inserted DataFrame into {table_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert DataFrame: {e}")
            return False


def create_supabase_manager_from_env() -> SupabaseManager:
    """
    Create SupabaseManager from environment variables.

    Environment variables:
    - SUPABASE_URL: Supabase project URL
    - SUPABASE_KEY: Supabase anon/service role key
    - SUPABASE_DB_URL: Direct PostgreSQL connection URL (optional)

    Returns:
        SupabaseManager instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
        )

    config = SupabaseConfig(url=url, key=key, db_url=os.getenv("SUPABASE_DB_URL"))

    return SupabaseManager(config)


def create_postgresql_manager_from_env() -> PostgreSQLManager:
    """
    Create PostgreSQLManager from environment variables.

    Environment variables:
    - DATABASE_URL or SUPABASE_DB_URL: PostgreSQL connection string

    Returns:
        PostgreSQLManager instance
    """
    connection_string = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")

    if not connection_string:
        raise ValueError(
            "DATABASE_URL or SUPABASE_DB_URL environment variable must be set"
        )

    return PostgreSQLManager(connection_string)
