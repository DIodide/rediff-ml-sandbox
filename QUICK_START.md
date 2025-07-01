# 🚀 Quick Start Guide

## 1. Set Up Environment

```bash
# Option A: Use the provided script <works>
./start_environment.sh

# Option B: Manual setup
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
pip install -r requirements.txt
```

## 2. Start Exploring (2 minutes)

```bash
# Activate environment if not already active
source ml_env/bin/activate

# Launch Jupyter Lab
jupyter lab
```

## 3. Try These First

### Graph ML

- Open `notebooks/01_graph_basics.ipynb`
- Run all cells to see graph creation, visualization, and GNN training

### Vector ML

- Open `notebooks/02_vector_ml_basics.ipynb`
- Explore FAISS similarity search and vector operations

### Database Integrations

- Open `notebooks/03_pinecone_integration.ipynb`
- Set up Pinecone vector database for production-scale similarity search
- Open `notebooks/04_supabase_integration.ipynb`
- Connect to Supabase PostgreSQL for structured data storage
- Open `notebooks/05_end_to_end_pipeline.ipynb`
- See complete pipeline combining graphs, vectors, and databases

## 4. Key Libraries Available

| Purpose             | Libraries                        |
| ------------------- | -------------------------------- |
| **Graph ML**        | PyTorch Geometric, NetworkX, DGL |
| **Vector Search**   | FAISS, Pinecone, Annoy, HNSWLIB  |
| **Databases**       | Supabase, SQLAlchemy, PostgreSQL |
| **Deep Learning**   | PyTorch, TensorFlow              |
| **Visualization**   | Matplotlib, Plotly, PyVis        |
| **Text Embeddings** | Sentence Transformers            |

## 5. Project Structure

```
📁 data/           # Your datasets
📁 notebooks/      # Jupyter experiments
📁 src/utils/      # Reusable functions
📁 experiments/    # Experiment tracking
📁 configs/        # Configuration files
```

## 6. Next Steps

1. **Load your data** into `data/raw/`
2. **Create experiments** in `notebooks/`
3. **Build reusable code** in `src/`
4. **Track results** in `experiments/`

## Need Help?

- Read the full [README.md](README.md)
- Database setup: [DATABASE_INTEGRATION_GUIDE.md](DATABASE_INTEGRATION_GUIDE.md)
- Check utility functions in `src/utils/`
- Explore the example notebooks
