import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np
from rdkit.Chem import AllChem, MolFromSmiles


@dataclass
class AddAndQueryResults:
    add_time: float
    query_time: float
    nn_ids: np.ndarray


@time
def add_and_query_index(
    index_vecs: np.ndarray, query_vecs: np.ndarray, index, num_neighbors: int = 20
) -> AddAndQueryResults:
    # Add vectors to index
    t0 = time.time()
    index.add(index_vecs)
    add_time = time.time() - t0

    # Query index
    t0 = time.time()
    distances, ids = index.search(query_vecs, num_neighbors)
    query_time = time.time() - t0

    return (
        AddAndQueryResults(add_time=add_time, query_time=query_time, nn_ids=ids),
        index,
    )


def morgan_fingerprint_embedding(smiles: str, radius: int, embed_dim: int):
    fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=embed_dim)
    mol = MolFromSmiles(smiles)
    return fp_gen.GetFingerprintAsNumPy(mol)


def embed_many_smiles(all_smiles: set, embed_fcn: Callable, **kwargs) -> np.ndarray:
    data = [embed_fcn(smiles, **kwargs) for smiles in all_smiles]
    return np.array(data)
