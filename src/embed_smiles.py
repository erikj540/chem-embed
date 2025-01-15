import gzip
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import click
import faiss
import numpy as np
from joblib import dump
from rdkit.Chem import AllChem, MACCSkeys, MolFromSmiles
from sklearn.model_selection import train_test_split

from src.constants import DATA_DIR, EMBEDDINGS_DIR
from src.utils import decompress_and_unpickle, pickle_and_compress


def morgan_fingerprint_embedding(smiles: str, radius: int, embed_dim: int):
    fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=embed_dim)
    mol = MolFromSmiles(smiles)
    return fp_gen.GetFingerprintAsNumPy(mol)


@click.command(
    help="This script embeds SMILES using a Morgan fingerprint.",
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.option(
    "--radius",
    "-r",
    type=int,
    help="The Morgan radius.",
)
@click.option(
    "--embed_dim",
    "-e",
    type=int,
    default=1024,
    help=f"Dimension of embedding space. Defaults to 1024",
)
@click.option("--smiles_path", "-s", type=str, help="Path to pickled SMILES")
@click.option("--testing", "-t", is_flag=True, help="Testing run", default=False)

# @click.option(
#     "--out_dir",
#     "-o",
#     type=str,
#     default=str(DATA_DIR),
#     help=f"Directory where to save the output file. Defaults to {DATA_DIR}",
# )
def embed_smiles_via_morgan_finterprint(
    radius: int,
    embed_dim: int,
    # out_dir: str,
    smiles_path: str,
    testing: bool,
):
    smiles_path = Path(smiles_path)
    out_path = (
        EMBEDDINGS_DIR / f"{smiles_path.stem}_morgan_radius={radius}_dim={embed_dim}"
    )
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        all_smiles = decompress_and_unpickle(smiles_path)

        # If this is a test run, reduce number of smiles and change
        # name of output file
        if testing:
            print("Testing run...")
            all_smiles = set(list(all_smiles)[:100])
            out_path = Path(str(out_path) + "_TEST")

        # Embed SMILES
        print(f"Embedding with dim={embed_dim}, radius={radius}...")
        t0 = time.time()
        data = []
        for smiles in all_smiles:
            try:
                data.append(
                    morgan_fingerprint_embedding(
                        smiles=smiles, radius=radius, embed_dim=embed_dim
                    )
                )
            except:
                continue
        print(f"\t Took {round(time.time()-t0, 2)} seconds")

        # Save embedding
        print(f"Saving embedding...")
        t0 = time.time()
        np.save(out_path, data)
        print(f"\t Took {round(time.time()-t0, 2)} seconds")


if __name__ == "__main__":
    embed_smiles_via_morgan_finterprint()

    # radii = [0, 1, 2, 3, 4]
    # dims = [1024, 2048]
