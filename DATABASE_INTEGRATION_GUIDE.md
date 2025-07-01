# Database Integration Guide

This guide will help you set up and use **Pinecone Vector Database** and **Supabase PostgreSQL** with your ML sandbox for production-ready vector search and data storage.

## 🗂️ Overview

The ML sandbox now includes modular database integrations:

- **Pinecone**: Managed vector database for similarity search at scale
- **Supabase**: PostgreSQL database with REST API, real-time features, and authentication
- **Hybrid Architecture**: Combine structured data (Supabase) with vector search (Pinecone)

## 🚀 Quick Setup

### 1. Environment Configuration

Copy the environment template and fill in your credentials:

```bash
cp env.template .env
```

Edit `.env` with your actual credentials:

```bash
# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=ml-sandbox-index

# Supabase
SUPABASE_URL=https://your_project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_DB_URL=postgresql://postgres:[PASSWORD]@db.your_project.supabase.co:5432/postgres
```

### 2. Install Dependencies

The required packages are already in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 📊 Pinecone Vector Database

### Getting Started with Pinecone

1. **Sign up** at [Pinecone.io](https://www.pinecone.io/)
2. **Create a project** and get your API key
3. **Copy your API key** to the `.env` file

### Basic Usage

```python
from src.utils.pinecone_utils import create_pinecone_manager_from_env

# Initialize from environment variables
pc_manager = create_pinecone_manager_from_env()

# Create index (one-time setup)
pc_manager.create_index(dimension=384, metric="cosine")
pc_manager.connect_to_index()

# Check index stats
stats = pc_manager.get_index_stats()
print(f"Index has {stats['total_vector_count']} vectors")
```

### Storing Vectors

```python
import pandas as pd
from sentence_transformers import SentenceTransformer

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = ["Machine learning is great", "AI will change the world", "Python is awesome"]
embeddings = model.encode(texts)

# Create DataFrame with vectors and metadata
df = pd.DataFrame({
    'id': [f'doc_{i}' for i in range(len(texts))],
    'text': texts,
    'vector': embeddings.tolist(),
    'category': ['tech', 'ai', 'programming']
})

# Upsert to Pinecone
success = pc_manager.upsert_vectors(
    df=df,
    id_column='id',
    vector_column='vector',
    metadata_columns=['text', 'category']
)
```

### Querying Vectors

```python
# Search for similar vectors
query_text = "What is artificial intelligence?"
query_embedding = model.encode([query_text])

results = pc_manager.query_vectors(
    query_vector=query_embedding[0],
    top_k=5,
    include_metadata=True,
    filter_dict={"category": "ai"}  # Optional metadata filter
)

# Process results
for match in results['matches']:
    print(f"Score: {match['score']:.3f}")
    print(f"Text: {match['metadata']['text']}")
    print(f"Category: {match['metadata']['category']}")
    print("---")
```

### Advanced Pinecone Features

```python
# Batch operations
vectors = [
    ("id1", [0.1, 0.2, 0.3], {"type": "document"}),
    ("id2", [0.4, 0.5, 0.6], {"type": "query"}),
]
pc_manager.upsert_vectors(vectors=vectors, batch_size=100)

# Delete vectors
pc_manager.delete_vectors(ids=["id1", "id2"])

# Namespace support (for multi-tenant applications)
results = pc_manager.query_vectors(
    query_vector=query_embedding[0],
    top_k=5,
    namespace="user_123"  # Isolate vectors by namespace
)
```

## 🗃️ Supabase PostgreSQL Database

### Getting Started with Supabase

1. **Sign up** at [Supabase.com](https://supabase.com/)
2. **Create a new project**
3. **Get your URL and anon key** from Settings > API
4. **Get your database URL** from Settings > Database

### Basic Usage

```python
from src.utils.supabase_utils import create_supabase_manager_from_env

# Initialize from environment variables
sb_manager = create_supabase_manager_from_env()

# Test connection
try:
    tables = sb_manager.query_data("information_schema.tables", limit=5)
    print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")
```

### Creating Tables

First, create tables in your Supabase dashboard or via SQL:

```sql
-- Example: ML experiments table
CREATE TABLE experiments (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    model_type TEXT,
    accuracy FLOAT,
    parameters JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Example: Documents table for metadata
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    title TEXT,
    content TEXT,
    category TEXT,
    embedding_id TEXT,  -- Reference to Pinecone vector
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Inserting Data

```python
# Insert single record
experiment_data = {
    "name": "GNN Node Classification",
    "model_type": "GraphSAGE",
    "accuracy": 0.8934,
    "parameters": {"hidden_dim": 64, "num_layers": 3}
}

result = sb_manager.insert_data("experiments", experiment_data)
print(f"Inserted experiment with ID: {result[0]['id']}")

# Bulk insert from DataFrame
import pandas as pd

df = pd.DataFrame({
    'name': ['Experiment 1', 'Experiment 2'],
    'model_type': ['GCN', 'GAT'],
    'accuracy': [0.85, 0.87]
})

success = sb_manager.bulk_insert_dataframe(df, "experiments")
```

### Querying Data

```python
# Simple queries
experiments = sb_manager.query_data("experiments")
print(f"Total experiments: {len(experiments)}")

# Filtered queries
high_accuracy = sb_manager.query_data(
    "experiments",
    columns=["name", "accuracy", "created_at"],
    filters={"accuracy": {"gte": 0.9}},
    order_by="-accuracy",
    limit=10
)

# Complex filters
recent_gnn_experiments = sb_manager.query_data(
    "experiments",
    filters={
        "model_type": {"like": "%GNN%"},
        "created_at": {"gte": "2024-01-01"}
    }
)

# Convert to DataFrame for analysis
df = sb_manager.query_to_dataframe("experiments")
print(df.describe())
```

### Updating and Deleting Data

```python
# Update records
updated = sb_manager.update_data(
    "experiments",
    data={"accuracy": 0.91},
    filters={"id": 1}
)

# Delete records
deleted = sb_manager.delete_data(
    "experiments",
    filters={"accuracy": {"lt": 0.5}}
)
```

## 🔄 Hybrid Architecture Patterns

### Pattern 1: Metadata in Supabase, Vectors in Pinecone

```python
# Store document metadata in Supabase
document_data = {
    "id": "doc_123",
    "title": "ML Research Paper",
    "content": "Full text content...",
    "category": "research"
}
sb_manager.insert_data("documents", document_data)

# Store vector in Pinecone with reference
vector_data = [
    ("doc_123", embedding.tolist(), {"title": "ML Research Paper", "category": "research"})
]
pc_manager.upsert_vectors(vectors=vector_data)

# Search: Vector similarity + metadata enrichment
search_results = pc_manager.query_vectors(query_embedding, top_k=10)

# Enrich with full metadata from Supabase
doc_ids = [match['id'] for match in search_results['matches']]
full_metadata = sb_manager.query_data(
    "documents",
    filters={"id": {"in": doc_ids}}
)
```

### Pattern 2: Experiment Tracking

```python
from datetime import datetime

# Start experiment
experiment = {
    "name": "Vector Search Evaluation",
    "status": "running",
    "started_at": datetime.now().isoformat()
}
exp_result = sb_manager.insert_data("experiments", experiment)
exp_id = exp_result[0]['id']

# Store experiment vectors in Pinecone namespace
namespace = f"exp_{exp_id}"
pc_manager.upsert_vectors(
    vectors=experiment_vectors,
    namespace=namespace
)

# Update experiment with results
sb_manager.update_data(
    "experiments",
    data={
        "status": "completed",
        "accuracy": final_accuracy,
        "completed_at": datetime.now().isoformat()
    },
    filters={"id": exp_id}
)
```

### Pattern 3: Real-time Recommendations

```python
# User interaction logging
user_interaction = {
    "user_id": "user_456",
    "item_id": "item_789",
    "interaction_type": "view",
    "timestamp": datetime.now().isoformat()
}
sb_manager.insert_data("user_interactions", user_interaction)

# Get user embedding from recent interactions
user_items = sb_manager.query_data(
    "user_interactions",
    filters={"user_id": "user_456"},
    order_by="-timestamp",
    limit=10
)

# Generate user embedding and search similar items
user_embedding = generate_user_embedding(user_items)
recommendations = pc_manager.query_vectors(
    user_embedding,
    top_k=20,
    filter_dict={"available": True}
)
```

## ⚙️ Configuration Management

### Using YAML Configuration

Create `configs/my_config.yaml`:

```yaml
pinecone:
  api_key: "${PINECONE_API_KEY}"
  index_name: "production-index"
  dimension: 1536
  metric: "cosine"

supabase:
  url: "${SUPABASE_URL}"
  key: "${SUPABASE_KEY}"
  schema: "public"

batch_settings:
  default_batch_size: 500
  max_batch_size: 5000
```

Load configuration:

```python
from src.utils.config_utils import setup_config

config = setup_config("my_config.yaml")
print(f"Using index: {config.pinecone.index_name}")
```

## 🚀 Production Best Practices

### 1. Connection Pooling

```python
# For high-throughput applications
sb_manager.setup_sqlalchemy_engine()

# Use connection pooling for batch operations
with sb_manager.engine.connect() as conn:
    # Bulk operations
    pass
```

### 2. Error Handling

```python
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

try:
    # Database operations
    result = sb_manager.insert_data("table", data)
except Exception as e:
    logging.error(f"Database operation failed: {e}")
    # Implement retry logic or fallback
```

### 3. Monitoring and Metrics

```python
# Monitor Pinecone usage
stats = pc_manager.get_index_stats()
print(f"Vector count: {stats['total_vector_count']}")
print(f"Index fullness: {stats['index_fullness']}")

# Monitor Supabase queries
query_start = time.time()
results = sb_manager.query_data("large_table")
query_time = time.time() - query_start
print(f"Query took {query_time:.2f} seconds")
```

### 4. Data Validation

```python
from pydantic import BaseModel, ValidationError

class ExperimentData(BaseModel):
    name: str
    accuracy: float
    model_type: str

# Validate before inserting
try:
    validated_data = ExperimentData(**raw_data)
    sb_manager.insert_data("experiments", validated_data.dict())
except ValidationError as e:
    print(f"Data validation failed: {e}")
```

## 🔧 Troubleshooting

### Common Issues

1. **Pinecone Connection Errors**

   ```python
   # Check API key and region
   import pinecone
   pinecone.init(api_key="your-key")
   print(pinecone.list_indexes())
   ```

2. **Supabase RLS Policies**

   ```sql
   -- Disable RLS for testing (enable in production)
   ALTER TABLE your_table DISABLE ROW LEVEL SECURITY;
   ```

3. **Vector Dimension Mismatches**
   ```python
   # Always check dimensions before upserting
   assert len(vector) == pc_manager.config.dimension
   ```

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger('src.utils').setLevel(logging.DEBUG)

# Test connections individually
from src.utils.pinecone_utils import PineconeConfig, PineconeManager
from src.utils.supabase_utils import SupabaseConfig, SupabaseManager

# Test configs
pc_config = PineconeConfig(api_key="test-key")
sb_config = SupabaseConfig(url="test-url", key="test-key")
```

## 📚 Next Steps

1. **Explore the Example Notebooks**:

   - `notebooks/03_pinecone_integration.ipynb`
   - `notebooks/04_supabase_integration.ipynb`
   - `notebooks/05_end_to_end_pipeline.ipynb`

2. **Scale Your Setup**:

   - Use Pinecone namespaces for multi-tenancy
   - Implement Supabase Row Level Security (RLS)
   - Set up monitoring and alerting

3. **Advanced Features**:
   - Real-time subscriptions with Supabase
   - Pinecone sparse-dense hybrid search
   - Vector metadata filtering optimization

## 🆘 Support

- **Pinecone**: [Documentation](https://docs.pinecone.io/) | [Community](https://community.pinecone.io/)
- **Supabase**: [Documentation](https://supabase.com/docs) | [Discord](https://discord.supabase.com/)
- **Project Issues**: [GitHub Issues](https://github.com/your-repo/issues)

Happy building! 🚀
