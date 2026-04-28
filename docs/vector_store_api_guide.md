---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

### Vector Store User Guide

Vector Store is a component designed for storing, managing, and retrieving vector embeddings. It supports features such as workspace management, similarity search, and metadata filtering.

## Core Concepts

**Workspace**: Each workspace is an independent vector storage unit used to organize and manage related vector nodes.

**VectorNode**: A data unit containing text content, vector embedding, and metadata. It serves as the fundamental unit for storage and retrieval.

**Embedding Model**: Used to convert text into vector embeddings. It supports automatic generation of both node vectors and query vectors.

## Available Implementations

FlowLLM provides multiple Vector Store implementations tailored to different use cases:

- **LocalVectorStore** ([source code](https://github.com/flowllm-ai/flowllm/blob/main/flowllm/core/vector_store/local_vector_store.py)): A file-based local implementation that persists data in JSONL format. Suitable for single-machine deployments and small-scale datasets.
- **MemoryVectorStore** ([source code](https://github.com/flowllm-ai/flowllm/blob/main/flowllm/core/vector_store/memory_vector_store.py)): An in-memory implementation offering fast access speeds. Ideal for temporary data or testing scenarios.
- **QdrantVectorStore** ([source code](https://github.com/flowllm-ai/flowllm/blob/main/flowllm/core/vector_store/qdrant_vector_store.py)): Built on the Qdrant vector database, supporting high-performance vector search. Recommended for large-scale production environments.
- **ChromaVectorStore** ([source code](https://github.com/flowllm-ai/flowllm/blob/main/flowllm/core/vector_store/chroma_vector_store.py)): Based on ChromaDB, providing persistent storage and metadata filtering capabilities.
- **EsVectorStore** ([source code](https://github.com/flowllm-ai/flowllm/blob/main/flowllm/core/vector_store/es_vector_store.py)): Built on Elasticsearch, enabling powerful combined full-text and vector search functionalities.
- **ObVecVectorStore** ([source code](https://github.com/agentscope-ai/ReMe/blob/main/reme/core/vector_store/obvec_vector_store.py)): Uses [pyobvector](https://pypi.org/project/pyobvector/) against **OceanBase** or **seekdb** (MySQL-compatible wire protocol). Suitable when you already run OceanBase/seekdb or need a SQL-native vector table with HNSW-style ANN search and JSON metadata filters.
- **ZvecVectorStore** ([source code](https://github.com/agentscope-ai/ReMe/blob/main/reme/core/vector_store/zvec_vector_store.py)): Built on zvec, a high-performance local vector database with strong-schema support and HNSW indexing. Suitable for single-machine deployments requiring fast vector search.

All Vector Store implementations inherit from **BaseVectorStore** ([source code](https://github.com/agentscope-ai/ReMe/blob/main/reme/core/vector_store/base_vector_store.py)) in ReMe, ensuring a consistent interface specification.

## Core Features

### Workspace Management

- **Create Workspace**: Create a new workspace for storing vector nodes.
- **Delete Workspace**: Remove a workspace along with all its data.
- **Check Workspace Existence**: Verify whether a specified workspace exists.
- **List Workspaces**: Retrieve a list of all existing workspaces.
- **Copy Workspace**: Duplicate data from one workspace to another.

### Node Operations

- **Insert Nodes**: Insert vector nodes into a workspace, supporting single or batch insertion with automatic vector embedding generation.
- **Delete Nodes**: Remove specific nodes by their IDs.
- **Iterate Nodes**: Traverse all nodes within a workspace.

### Vector Search

- **Similarity Search**: Perform vector similarity searches based on text queries, returning the top-K most similar results.
- **Metadata Filtering**: Apply filtering conditions based on metadata, including exact matches and range queries.
- **Similarity Scores**: Search results include similarity scores to evaluate match quality.

### Data Import/Export

- **Export Workspace**: Export workspace data to a file or specified path.
- **Import Workspace**: Import data into a workspace from a file or a list of nodes.
- **Callback Functions**: Support callback functions during import/export for data transformation.

## Synchronous and Asynchronous Interfaces

All Vector Store implementations provide both synchronous and asynchronous interfaces:

- **Synchronous Interface**: Direct method calls suitable for synchronous code environments.
- **Asynchronous Interface**: Prefixed with `async_`, designed for asynchronous environments and offering better concurrency performance.

The asynchronous interface is particularly useful in the following scenarios:
- Using asynchronous embedding models for vector generation.
- Performing batch operations in high-concurrency environments.
- Integrating with other asynchronous components.

## Configuration Options

### General Configuration

- **embedding_model**: Instance of the embedding model used to generate vector embeddings.
- **batch_size**: Batch size for bulk operations (default: 1024).

### LocalVectorStore Configuration

- **store_dir**: Storage directory path (default: `./local_vector_store`).

### MemoryVectorStore Configuration

- **store_dir**: Persistence directory (default: `./memory_vector_store`).

### QdrantVectorStore Configuration

- **url**: Qdrant service URL (optional; used for Qdrant Cloud or custom deployments).
- **host**: Qdrant server host (default: `localhost`).
- **port**: Qdrant server port (default: `6333`).
- **api_key**: API key for Qdrant Cloud authentication.
- **distance**: Distance metric—supports COSINE, EUCLIDEAN, DOT (default: COSINE).

### ChromaVectorStore Configuration

- **store_dir**: ChromaDB data storage directory (default: `./chroma_vector_store`).

### EsVectorStore Configuration

- **hosts**: Elasticsearch host address(es), either a string or a list (default: `http://localhost:9200`).
- **basic_auth**: Basic authentication credentials (username and password).

### ObVecVectorStore Configuration

- **uri**: Server address as `host:port` (default: `127.0.0.1:2881`).
- **user**: MySQL-compatible user. seekdb single-tenant images often use `root`; OceanBase multi-tenant setups typically use `root@<tenant>` (e.g. `root@test`).
- **password**: Database password (seekdb Docker images commonly set this via `ROOT_PASSWORD`).
- **database**: Logical database name (default: `test`).
- **index_metric**: Distance metric for the vector index: `cosine` or `ip` (inner product); default `cosine`.
- **index_ef_search**: HNSW `ef_search` parameter passed to pyobvector (default: `100`).
- **collection_name**: Table name for the collection (from `VectorStoreConfig`, default `reme`). Use lowercase names if your deployment restricts identifiers.

**Local seekdb via Docker**

```text
docker run -d --name reme_seekdb -p 2881:2881 -e ROOT_PASSWORD=<your_root_password> quay.io/oceanbase/seekdb:latest
```

**Integration tests** (requires a running server, embedding API credentials in `.env`, and matching DB password):

```shell
OBVEC_PASSWORD=<your_root_password> python tests/test_vector_store.py --obvec
```
### ZvecVectorStore Configuration

- **db_path**: Local storage path for persistent mode (required).
- **dimension**: Dimensionality of the embedding vectors (default: `1024`).
- **distance**: Distance metric — supports `cosine`, `l2`, `ip` (default: `cosine`).

## Configuration File Examples

Configure Vector Store in `flowllm/config/default.yaml` under the `vector_store` section. The basic structure is as follows:

```yaml
vector_store:
  default:
    backend: <backend_name>        # Required: vector store backend type
    embedding_model: default       # Required: name of embedding model config
    params:                        # Optional: backend-specific parameters
      # Backend-specific parameters
```

```shell
vector_store.default.backend=<backend_name>
vector_store.default.params.<param_name>=<param_value>
```

### Configuration Field Descriptions

- **`backend`** (required): Vector store backend type. Options: `local`, `memory`, `chroma`, `qdrant`, `elasticsearch`, `obvec`, `zvec`.
- **`embedding_model`** (required): Name of the embedding model configuration, referencing the `embedding_model` section.
- **`params`** (optional): Dictionary of backend-specific parameters passed to the vector store constructor.

### Configuration Examples by Type

#### 1. LocalVectorStore Configuration

Simplest local file-based storage, ideal for development and testing.

**Implementation**: [`flowllm/core/vector_store/local_vector_store.py`](https://github.com/flowllm-ai/flowllm/blob/main/flowllm/core/vector_store/local_vector_store.py)

```yaml
vector_store:
  default:
    backend: local
    embedding_model: default
    params:
      store_dir: "./local_vector_store"  # Storage directory (optional; default: "./local_vector_store")
```

```shell
vector_store.default.backend=local
vector_store.default.params.store_dir=./local_vector_store
```

#### 2. MemoryVectorStore Configuration

In-memory storage with fast access, suitable for temporary data or testing.

**Implementation**: [`flowllm/core/vector_store/memory_vector_store.py`](https://github.com/flowllm-ai/flowllm/blob/main/flowllm/core/vector_store/memory_vector_store.py)

```yaml
vector_store:
  default:
    backend: memory
    embedding_model: default
    params:
      store_dir: "./memory_vector_store"  # Persistence directory (optional; default: "./memory_vector_store")
```

```shell
vector_store.default.backend=memory
vector_store.default.params.store_dir=./memory_vector_store
```

#### 3. ChromaVectorStore Configuration

Persistent storage based on ChromaDB with metadata filtering support.

**Implementation**: [`flowllm/core/vector_store/chroma_vector_store.py`](https://github.com/flowllm-ai/flowllm/blob/main/flowllm/core/vector_store/chroma_vector_store.py)

```yaml
vector_store:
  default:
    backend: chroma
    embedding_model: default
    params:
      store_dir: "./chroma_vector_store"  # ChromaDB data directory (optional; default: "./chroma_vector_store")
```

```shell
vector_store.default.backend=chroma
vector_store.default.params.store_dir=./chroma_vector_store
```

#### 4. QdrantVectorStore Configuration

**Implementation**: [`flowllm/core/vector_store/qdrant_vector_store.py`](https://github.com/flowllm-ai/flowllm/blob/main/flowllm/core/vector_store/qdrant_vector_store.py)

**Local Qdrant Instance**:

```yaml
vector_store:
  default:
    backend: qdrant
    embedding_model: default
    params:
      host: "localhost"      # Qdrant server host (optional; default: localhost)
      port: 6333             # Qdrant server port (optional; default: 6333)
      distance: "COSINE"     # Distance metric (optional; default: COSINE; options: COSINE, EUCLIDEAN, DOT)
```

```shell
vector_store.default.backend=qdrant
vector_store.default.params.host=localhost
vector_store.default.params.port=6333
vector_store.default.params.distance=COSINE
```

**Qdrant Cloud Configuration**:

```yaml
vector_store:
  default:
    backend: qdrant
    embedding_model: default
    params:
      url: "https://your-cluster.qdrant.io:6333"  # Qdrant Cloud URL
      api_key: "your-api-key-here"                 # API key
      distance: "COSINE"
```

```shell
vector_store.default.backend=qdrant
vector_store.default.params.url=https://your-cluster.qdrant.io:6333
vector_store.default.params.api_key=your-api-key-here
vector_store.default.params.distance=COSINE
```

#### 5. EsVectorStore Configuration

**Implementation**: [`flowllm/core/vector_store/es_vector_store.py`](https://github.com/flowllm-ai/flowllm/blob/main/flowllm/core/vector_store/es_vector_store.py)

**Basic Configuration (Local Elasticsearch)**:

```yaml
vector_store:
  default:
    backend: elasticsearch
    embedding_model: default
    params:
      hosts: "http://localhost:9200"  # Elasticsearch host(s) (optional; default: http://localhost:9200)
```

```shell
vector_store.default.backend=elasticsearch
vector_store.default.params.hosts=http://localhost:9200
```

**Configuration with Authentication**:

```yaml
vector_store:
  default:
    backend: elasticsearch
    embedding_model: default
    params:
      hosts: "http://elasticsearch.example.com:9200"
      basic_auth: ["username", "password"]  # Basic auth credentials
```

```shell
vector_store.default.backend=elasticsearch
vector_store.default.params.hosts=http://elasticsearch.example.com:9200
vector_store.default.params.basic_auth='["username", "password"]'
```

**Multi-Host Configuration**:

```yaml
vector_store:
  default:
    backend: elasticsearch
    embedding_model: default
    params:
      hosts:
        - "http://es-node1:9200"
        - "http://es-node2:9200"
        - "http://es-node3:9200"
```

```shell
vector_store.default.backend=elasticsearch
vector_store.default.params.hosts='["http://es-node1:9200", "http://es-node2:9200", "http://es-node3:9200"]'
```

#### 6. ObVecVectorStore Configuration (OceanBase / seekdb)

**Implementation**: [`reme/core/vector_store/obvec_vector_store.py`](https://github.com/agentscope-ai/ReMe/blob/main/reme/core/vector_store/obvec_vector_store.py)

**Example (seekdb on localhost)**:

```yaml
vector_stores:
  default:
    backend: obvec
    embedding_model: default
    collection_name: reme
    uri: "127.0.0.1:2881"
    user: "root"
    password: "your-root-password"
    database: "test"
    index_metric: "cosine"
    index_ef_search: 100
```

```shell
vector_stores.default.backend=obvec
vector_stores.default.uri=127.0.0.1:2881
vector_stores.default.user=root
vector_stores.default.password=your-root-password
```

ReMe service YAML uses the key `vector_stores` (plural); CLI overrides use the same nested paths.

#### 7. ZvecVectorStore Configuration

Persistent local storage based on zvec with HNSW indexing and strong-schema support.

**Implementation**: [`reme/core/vector_store/zvec_vector_store.py`](https://github.com/agentscope-ai/ReMe/blob/main/reme/core/vector_store/zvec_vector_store.py)

```yaml
vector_store:
  default:
    backend: zvec
    embedding_model: default
    params:
      db_path: "./zvec_vector_store"  # Local storage path (required)
      dimension: 1024                 # Vector dimension (optional; default: 1024)
      distance: "cosine"              # Distance metric (optional; default: cosine; options: cosine, l2, ip)
```

```shell
vector_store.default.backend=zvec
vector_store.default.params.db_path=./zvec_vector_store
vector_store.default.params.dimension=1024
vector_store.default.params.distance=cosine
```

### Complete Configuration Example

Below is a complete `default.yaml` example including both embedding model and vector store configurations:

```yaml
# Embedding model configuration
embedding_model:
  default:
    backend: openai_compatible
    model_name: text-embedding-v4
    params:
      dimensions: 1024

# Vector store configuration
vector_store:
  default:
    backend: elasticsearch
    embedding_model: default
    params:
      hosts: "http://localhost:9200"
```

```shell
# Embedding model configuration
embedding_model.default.backend=openai_compatible
embedding_model.default.model_name=text-embedding-v4
embedding_model.default.params.dimensions=1024

# Vector store configuration
vector_store.default.backend=elasticsearch
vector_store.default.params.hosts=http://localhost:9200
```

### Environment Variable Support

Certain Vector Stores support environment variables as a supplement to YAML configuration:

- **Elasticsearch**: `FLOW_ES_HOSTS` – Elasticsearch host address.
- **Qdrant**:
  - `FLOW_QDRANT_HOST` – Qdrant host (default: `localhost`)
  - `FLOW_QDRANT_PORT` – Qdrant port (default: `6333`)
  - `FLOW_QDRANT_API_KEY` – Qdrant API key

When parameters are not explicitly specified in the YAML configuration, the system falls back to environment variables.

## Metadata Filtering

Two types of metadata filtering are supported:

- **Exact Match**: Specify field values for exact matching.
- **Range Queries**: Use operators `gte`, `lte`, `gt`, `lt` for numeric range queries.
- **Nested Fields**: Access nested metadata fields using dot notation.

## Usage Recommendations

- **Development & Testing**: Use MemoryVectorStore or LocalVectorStore—no additional services required.
- **Small-Scale Applications**: Use LocalVectorStore or ChromaVectorStore for simplicity and ease of use.
- **Production Environments**: Use QdrantVectorStore, EsVectorStore, or ObVecVectorStore (OceanBase/seekdb) for high performance and scalability, depending on your existing infrastructure.
- **High-Performance Local Search**: Use ZvecVectorStore for single-machine deployments requiring fast HNSW-based vector search with local persistence.
- **Hybrid Search**: Use EsVectorStore to combine vector search with full-text search capabilities.
- **OceanBase / seekdb**: Use ObVecVectorStore when you standardize on pyobvector and SQL-accessible vector tables.

## Important Notes

- Ensure the embedding model’s output dimension matches the Vector Store configuration.
- For large-scale data, use professional vector databases (e.g., Qdrant, Elasticsearch).
- Asynchronous interfaces deliver better performance in asynchronous environments.
- Regularly back up critical data, especially when using in-memory storage.
- Choose an appropriate batch size based on your data scale to optimize performance.
