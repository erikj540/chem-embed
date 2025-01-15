import gzip
import pickle
from typing import Any


def pickle_and_compress(obj: Any, file_path: str):
    with gzip.open(file_path, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def decompress_and_unpickle(filename: str):
    with gzip.open(filename, "rb") as file:
        return pickle.load(file)


def get_chembl_smiles_file_name(chembl_version: str) -> str:
    return f"./chembl_v{chembl_version}_smiles.pkl"
