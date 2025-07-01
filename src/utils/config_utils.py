"""
Configuration management utilities.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
import logging

try:
    from dotenv import load_dotenv
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "Required packages not installed. Install with: pip install python-dotenv pydantic"
    )

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Pydantic model for database configuration validation."""

    class PineconeConfig(BaseModel):
        api_key: str
        environment: Optional[str] = None
        index_name: str = "ml-sandbox-index"
        dimension: int = 1536
        metric: str = "cosine"
        cloud: str = "aws"
        region: str = "us-east-1"

    class SupabaseConfig(BaseModel):
        url: str
        key: str
        db_url: Optional[str] = None
        schema: str = "public"

    class PostgresConfig(BaseModel):
        host: str = "localhost"
        port: int = 5432
        database: str = "ml_sandbox"
        username: str
        password: str

    class ConnectionPoolConfig(BaseModel):
        max_connections: int = 20
        min_connections: int = 5
        connection_timeout: int = 30

    class BatchConfig(BaseModel):
        default_batch_size: int = 1000
        max_batch_size: int = 10000
        concurrent_batches: int = 3

    class LoggingConfig(BaseModel):
        level: str = "INFO"
        log_to_file: bool = True
        log_file: str = "logs/database.log"

    pinecone: Optional[PineconeConfig] = None
    supabase: Optional[SupabaseConfig] = None
    postgres: Optional[PostgresConfig] = None
    connection_pool: ConnectionPoolConfig = Field(default_factory=ConnectionPoolConfig)
    batch_settings: BatchConfig = Field(default_factory=BatchConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_env_file(env_file: str = ".env", search_parent_dirs: bool = True) -> bool:
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to environment file
        search_parent_dirs: Whether to search parent directories

    Returns:
        True if file was loaded successfully
    """
    try:
        env_path = Path(env_file)

        # Search in parent directories if file not found
        if not env_path.exists() and search_parent_dirs:
            current_dir = Path.cwd()
            for parent in [current_dir] + list(current_dir.parents):
                potential_path = parent / env_file
                if potential_path.exists():
                    env_path = potential_path
                    break

        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from {env_path}")
            return True
        else:
            logger.warning(f"Environment file {env_file} not found")
            return False

    except Exception as e:
        logger.error(f"Failed to load environment file: {e}")
        return False


def load_yaml_config(
    config_file: str, substitute_env_vars: bool = True
) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_file: Path to YAML configuration file
        substitute_env_vars: Whether to substitute environment variables

    Returns:
        Configuration dictionary
    """
    try:
        config_path = Path(config_file)

        # Search in common config directories
        if not config_path.exists():
            search_paths = [
                Path.cwd() / config_file,
                Path.cwd() / "configs" / config_file,
                Path.cwd() / "config" / config_file,
            ]

            for path in search_paths:
                if path.exists():
                    config_path = path
                    break

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_path, "r") as f:
            config_text = f.read()

        # Substitute environment variables
        if substitute_env_vars:
            config_text = substitute_env_variables(config_text)

        config = yaml.safe_load(config_text)
        logger.info(f"Loaded configuration from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        raise


def substitute_env_variables(text: str) -> str:
    """
    Substitute environment variables in text using ${VAR_NAME} syntax.

    Args:
        text: Text containing environment variable references

    Returns:
        Text with substituted values
    """
    import re

    def replace_env_var(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) else ""
        return os.getenv(var_name, default_value)

    # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
    pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"
    return re.sub(pattern, replace_env_var, text)


def create_database_config(config_file: Optional[str] = None) -> DatabaseConfig:
    """
    Create and validate database configuration.

    Args:
        config_file: Optional path to YAML config file

    Returns:
        Validated DatabaseConfig instance
    """
    # Load environment variables
    load_env_file()

    # Start with default config
    config_dict = {}

    # Load from YAML file if provided
    if config_file:
        try:
            yaml_config = load_yaml_config(config_file)
            config_dict.update(yaml_config)
        except Exception as e:
            logger.warning(f"Could not load config file {config_file}: {e}")

    # Override with environment variables
    env_config = create_config_from_env()
    merge_dict(config_dict, env_config)

    # Validate and return
    try:
        return DatabaseConfig(**config_dict)
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def create_config_from_env() -> Dict[str, Any]:
    """
    Create configuration dictionary from environment variables.

    Returns:
        Configuration dictionary
    """
    config = {}

    # Pinecone configuration
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if pinecone_api_key:
        config["pinecone"] = {
            "api_key": pinecone_api_key,
            "environment": os.getenv("PINECONE_ENVIRONMENT"),
            "index_name": os.getenv("PINECONE_INDEX_NAME", "ml-sandbox-index"),
        }

    # Supabase configuration
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if supabase_url and supabase_key:
        config["supabase"] = {
            "url": supabase_url,
            "key": supabase_key,
            "db_url": os.getenv("SUPABASE_DB_URL"),
        }

    # PostgreSQL configuration
    db_url = os.getenv("DATABASE_URL")
    db_username = os.getenv("DB_USERNAME")
    db_password = os.getenv("DB_PASSWORD")

    if db_username and db_password:
        config["postgres"] = {
            "username": db_username,
            "password": db_password,
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME", "ml_sandbox"),
        }

    # Logging configuration
    log_level = os.getenv("LOG_LEVEL", "INFO")
    config["logging"] = {
        "level": log_level,
    }

    return config


def merge_dict(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> None:
    """
    Recursively merge dict2 into dict1.

    Args:
        dict1: Dictionary to merge into
        dict2: Dictionary to merge from
    """
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            merge_dict(dict1[key], value)
        else:
            dict1[key] = value


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "pinecone.api_key")
        default: Default value if key not found

    Returns:
        Configuration value
    """
    keys = key_path.split(".")
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def validate_required_env_vars(required_vars: List[str]) -> List[str]:
    """
    Validate that required environment variables are set.

    Args:
        required_vars: List of required environment variable names

    Returns:
        List of missing variables
    """
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    return missing_vars


def setup_logging(config: DatabaseConfig) -> None:
    """
    Set up logging based on configuration.

    Args:
        config: Database configuration
    """
    log_level = getattr(logging, config.logging.level.upper())

    # Create logs directory if needed
    if config.logging.log_to_file:
        log_path = Path(config.logging.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    handlers = [logging.StreamHandler()]

    if config.logging.log_to_file:
        handlers.append(logging.FileHandler(config.logging.log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


# Convenience function for quick setup
def setup_config(config_file: Optional[str] = "database_config.yaml") -> DatabaseConfig:
    """
    Set up configuration with logging.

    Args:
        config_file: Optional configuration file path

    Returns:
        Validated configuration
    """
    try:
        config = create_database_config(config_file)
        setup_logging(config)
        return config
    except Exception as e:
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO)
        logger.error(f"Failed to load configuration: {e}")
        raise
