"""
Configuration management for the RAG chatbot system.
Provides centralized configuration loading and validation.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    vector_db_provider: str
    vector_db_config: Dict[str, Any]
    metadata_db_provider: str
    metadata_db_config: Dict[str, Any]
    cache_provider: str
    cache_config: Dict[str, Any]


@dataclass
class LLMConfig:
    """Model configuration settings."""
    llm_config: Dict[str, Any]
    embedding_config: Dict[str, Any]
    performance_config: Dict[str, Any]


@dataclass
class EmbeddingConfig:
    """Embedding configuration settings."""
    provider: str
    model_name: str
    additional_params: Dict[str, Any] = None


@dataclass
class RetrievalConfig:
    """Retrieval configuration settings."""
    retrieval_config: Dict[str, Any]
    query_processing_config: Dict[str, Any]


@dataclass
class IngestionConfig:
    """Data ingestion configuration settings."""
    processors: Dict[str, Any]
    text_processing: Dict[str, Any]
    web_crawling: Dict[str, Any]
    batch_processing: Dict[str, Any]
    error_handling: Dict[str, Any]
    deduplication: Dict[str, Any]


@dataclass
class AppConfig:
    """Application configuration settings."""
    api_config: Dict[str, Any]
    chat_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    environment: str
    features: Dict[str, bool]


class ConfigManager:
    """Centralized configuration manager for the RAG chatbot system."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Path to configuration directory. Defaults to 'configs'
        """
        self.config_dir = Path(config_dir or r"E:\rag-chat-bot\src\configs")
        self._configs = {}
        self._load_all_configs()
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Replace environment variables
                content = self._replace_env_vars(content)
                return yaml.safe_load(content)
        except Exception as e:
            print(f"Error loading config file {file_path}: {e}")
            return {}
    
    def _replace_env_vars(self, content: str) -> str:
        """Replace environment variables in config content."""
        import re
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default_value)
        
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        return re.sub(pattern, replace_var, content)
    
    def _load_all_configs(self):
        """Load all configuration files."""
        config_files = {
            'database': 'database.yaml',
            'llms': 'llms.yaml',
            'embedding': 'embedding.yaml',
            'retrieval': 'retrieval.yaml',
            'ingestion': 'ingestion.yaml',
            'app': 'app.yaml'
        }
        
        for config_name, filename in config_files.items():
            file_path = self.config_dir / filename
            if file_path.exists():
                self._configs[config_name] = self._load_yaml_file(file_path)
                print(f"Loaded {config_name} configuration")
            else:
                print(f"Configuration file not found: {file_path}")
                self._configs[config_name] = {}
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        config = self._configs.get('database', {})
        
        return DatabaseConfig(
            vector_db_provider=config.get('vector_db', {}).get('provider', 'chroma'),
            vector_db_config=config.get('vector_db', {}),
            metadata_db_provider=config.get('metadata_db', {}).get('provider', 'sqlite'),
            metadata_db_config=config.get('metadata_db', {}),
            cache_provider=config.get('cache', {}).get('provider', 'memory'),
            cache_config=config.get('cache', {})
        )
    
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        config = self._configs.get('llms', {})
        
        return LLMConfig(
            llm_config=config.get('llm', {}),
            embedding_config=config.get('embedding', {}),
            performance_config=config.get('performance', {})
        )

    def get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        config = self._configs.get('embedding', {}).get('embedding', {})
        
        return EmbeddingConfig(
            provider=config.get('provider', 'sentence-transformer'),
            model_name=config.get('model_name', 'all-MiniLM-L6-v2'),
            additional_params=config.get('additional_params', {})
        )
    
    def get_retrieval_config(self) -> RetrievalConfig:
        """Get retrieval configuration."""
        config = self._configs.get('retrieval', {})
        
        return RetrievalConfig(
            retrieval_config=config.get('retrieval', {}),
            query_processing_config=config.get('query_processing', {})
        )
    
    def get_ingestion_config(self) -> IngestionConfig:
        """Get ingestion configuration."""
        config = self._configs.get('ingestion', {})
        
        return IngestionConfig(
            processors=config.get('ingestion', {}).get('processors', {}),
            text_processing=config.get('ingestion', {}).get('text_processing', {}),
            web_crawling=config.get('ingestion', {}).get('web_crawling', {}),
            batch_processing=config.get('ingestion', {}).get('batch_processing', {}),
            error_handling=config.get('ingestion', {}).get('error_handling', {}),
            deduplication=config.get('ingestion', {}).get('deduplication', {})
        )
    
    def get_app_config(self) -> AppConfig:
        """Get application configuration."""
        config = self._configs.get('app', {})
        
        return AppConfig(
            api_config=config.get('api', {}),
            chat_config=config.get('chat', {}),
            monitoring_config=config.get('monitoring', {}),
            environment=config.get('environment', {}).get('name', 'development'),
            features=config.get('features', {})
        )
    
    def get_config(self, config_name: str, default: Any = None) -> Any:
        """Get a specific configuration value by path."""
        try:
            keys = config_name.split('.')
            current = self._configs
            
            for key in keys:
                current = current[key]
            
            return current
        except (KeyError, TypeError):
            return default
    
    def reload_configs(self):
        """Reload all configuration files."""
        self._configs.clear()
        self._load_all_configs()
        print("Configuration reloaded")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    return config_manager


def reload_config():
    """Reload the global configuration."""
    global config_manager
    config_manager.reload_configs()
