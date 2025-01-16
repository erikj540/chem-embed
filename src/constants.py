from pathlib import Path

GIT_REPO_DIR = Path(__file__).parents[1]
DATA_DIR = GIT_REPO_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
INDICES_DIR = DATA_DIR / "indices"
