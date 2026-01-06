"""
Database migrations for Huxley.

Provides automatic table creation and vector extension setup
for supported database providers.
"""

from huxley.memory.migrations.schema import (
    get_base_schema,
    get_vector_schema,
    setup_database,
    setup_vectors,
    test_connection,
    setup_database_sync,
    test_connection_sync,
)

__all__ = [
    "get_base_schema",
    "get_vector_schema", 
    "setup_database",
    "setup_vectors",
    "test_connection",
    "setup_database_sync",
    "test_connection_sync",
]
