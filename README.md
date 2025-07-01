# Rediff Sandbox

A Python sandbox environment for graph-based and vector-based machine learning experiments meant to develop and test the utility to meet the functional requirements of ReDiff/Drift.
Things to explore include notions of graph and vector similarity, as well as representations
of code in either of these two formats.

## 🚀 Quick Start

### 1. Set up Python Environment

```bash
# Create virtual environment
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Jupyter Lab

```bash
jupyter lab
```

## 📁 Project Structure

```
rediff-ml-sandbox/
├── data/
│   ├── raw/           # Original, immutable data
│   ├── processed/     # Cleaned and preprocessed data
│   └── external/      # External datasets and references
├── notebooks/         # Jupyter notebooks for exploration
├── src/
│   ├── models/        # Model definitions and training scripts
│   ├── utils/         # Utility functions and helpers
│   └── data/          # Data loading and preprocessing
├── tests/             # Unit tests
├── configs/           # Configuration files
├── experiments/       # Experiment tracking and results
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🛠 Key Libraries Included

### Graph Machine Learning

- **PyTorch Geometric**: Graph neural networks and geometric deep learning
- **DGL**: Deep Graph Library for scalable GNNs
- **NetworkX**: Graph creation, manipulation, and analysis
- **StellarGraph**: Machine learning on graphs

### Vector Operations & Similarity Search

- **FAISS**: Efficient similarity search and clustering
- **Pinecone**: Managed vector database for production workloads
- **Annoy**: Approximate nearest neighbors
- **HNSWLIB**: Fast approximate nearest neighbor search
- **Sentence Transformers**: State-of-the-art text embeddings

### Database Integrations

- **Supabase**: PostgreSQL database with REST API and real-time features
- **SQLAlchemy**: Python SQL toolkit and ORM
- **Asyncpg/Psycopg2**: High-performance PostgreSQL adapters

### Core ML Stack

- **PyTorch & TensorFlow**: Deep learning frameworks
- **Scikit-learn**: Traditional machine learning
- **NumPy, Pandas, SciPy**: Data manipulation and scientific computing

### Visualization

- **Matplotlib, Seaborn, Plotly**: Static and interactive plots
- **PyVis**: Interactive network visualization
- **Bokeh**: Interactive web-based visualizations

### Database Integration

```python
# Pinecone vector database
from src.utils.pinecone_utils import create_pinecone_manager_from_env

pc_manager = create_pinecone_manager_from_env()
pc_manager.create_index(dimension=384)
pc_manager.connect_to_index()

# Store vectors
vectors = [(f"doc_{i}", embedding.tolist(), {"category": "ML"})
           for i, embedding in enumerate(embeddings)]
pc_manager.upsert_vectors(vectors)

# Search similar vectors
results = pc_manager.query_vectors(query_embedding, top_k=5)
```

```python
# Supabase database
from src.utils.supabase_utils import create_supabase_manager_from_env

sb_manager = create_supabase_manager_from_env()

# Insert data
data = {"name": "ML Experiment", "accuracy": 0.95, "created_at": "2024-01-01"}
sb_manager.insert_data("experiments", data)

# Query data with filters
results = sb_manager.query_data(
    "experiments",
    filters={"accuracy": {"gte": 0.9}},
    order_by="-created_at"
)
```

## 🔧 Development Setup

### Code Quality Tools

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy src/

# Run tests
pytest tests/
```

### GPU Support (Optional)

If you have CUDA available, uncomment the GPU-specific packages in `requirements.txt`:

```
# torch-geometric>=2.3.0+cu118
# faiss-gpu>=1.7.4
```

## 📊 Experiment Tracking

The `experiments/` directory is set up for tracking your ML experiments. Consider integrating with:

- **Weights & Biases (wandb)**: For experiment tracking
- **MLflow**: For ML lifecycle management
- **TensorBoard**: For visualization

## Contributing

1. Keep notebooks in the `notebooks/` directory
2. Put reusable code in `src/`
3. Add tests for new functionality in `tests/`
4. Update this README when adding new major components

## Resources

### Graph ML Resources

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [DGL User Guide](https://docs.dgl.ai/)
- [Graph Neural Networks Course](https://web.stanford.edu/class/cs224w/)

### Vector ML Resources

- [FAISS Documentation](https://faiss.ai/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Vector Similarity Search Guide](https://www.pinecone.io/learn/what-is-similarity-search/)

### Database Resources

- [Supabase Documentation](https://supabase.com/docs)
- [SQLAlchemy Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
