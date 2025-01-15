import gzip
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np
from rdkit.Chem import AllChem, MACCSkeys, MolFromSmiles
from sklearn.model_selection import train_test_split

from utils import decompress_and_unpickle, pickle_and_compress


@dataclass
class AddAndQueryResults:
    add_time: float
    query_time: float
    nn_ids: np.ndarray


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

    return AddAndQueryResults(add_time=add_time, query_time=query_time, nn_ids=ids)


def morgan_fingerprint_embedding(smiles: str, radius: int, embed_dim: int):
    fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=embed_dim)
    mol = MolFromSmiles(smiles)
    return fp_gen.GetFingerprintAsNumPy(mol)


def embed_many_smiles(all_smiles: set, embed_fcn: Callable, **kwargs) -> np.ndarray:
    data = [embed_fcn(smiles, **kwargs) for smiles in all_smiles]
    return np.array(data)


def ecfp_experiment(
    radius: int,
    embed_dim: int,
    smiles_path: str,
    rand_seed: int = 42069,
    num_query_vecs: int = 1000,
    num_neighbors: int = 20,
):
    # Define indices to test
    m_vals = [1, 5, 10, 20]
    indices = {"flat": faiss.IndexFlatL2(embed_dim)}
    for m in m_vals:
        indices[f"hnsw-{m}"] = faiss.IndexHNSWFlat(embed_dim, m)

    # Embed data
    print("Embedding SMILES...")
    t0 = time.time()
    smiles = decompress_and_unpickle(smiles_path)
    smiles = set(list(smiles)[:1001])
    data = embed_many_smiles(
        all_smiles=smiles,
        embed_fcn=morgan_fingerprint_embedding,
        radius=radius,
        embed_dim=embed_dim,
    )
    print(f"\t Took {round(time.time()-t0, 2)} seconds")

    # Split data into vectors that will go into the index and those that we'll use to
    # query the index
    index_vecs, query_vecs = train_test_split(
        data, test_size=num_query_vecs, random_state=rand_seed
    )

    # Add vectors to index and query it
    results = {}
    for name, index in indices.items():
        print(f"Running index {name}...")
        t0 = time.time()
        results[name] = add_and_query_index(
            index_vecs=index_vecs,
            query_vecs=query_vecs,
            index=index,
            num_neighbors=num_neighbors,
        )
        print(f"\t Took {round(time.time()-t0, 2)} seconds")

    pickle_and_compress(results, f"./exp_radius={radius}_embedDim={embed_dim}.pkl")


if __name__ == "__main__":
    smiles_path = Path("./chembl_v23_smiles.pkl")
    # ecfp_experiment(embed_dim=1024, radius=0, smiles_path=smiles_path)

    radii = [0, 1, 2, 3, 4]
    dims = [1024, 2048]
    smiles = decompress_and_unpickle(smiles_path)
    for dim in dims:
        for radius in radii:
            print(f"Embedding with dim={dim}, radius={radius}...")
            t0 = time.time()
            data = embed_many_smiles(
                all_smiles=smiles,
                embed_fcn=morgan_fingerprint_embedding,
                radius=radius,
                embed_dim=dim,
            )
            print(f"\t Took {round(time.time()-t0, 2)} seconds")

            output_file = f"./{smiles_path.stem}_radius={radius}_dim={dim}.pkl"
            pickle_and_compress(data, output_file)
