import gzip
import pickle
import time
from typing import Any


def pickle_and_compress(obj: Any, file_path: str):
    with gzip.open(file_path, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def decompress_and_unpickle(filename: str):
    with gzip.open(filename, "rb") as file:
        return pickle.load(file)


def get_chembl_smiles_file_name(chembl_version: str) -> str:
    return f"./chembl_v{chembl_version}_smiles.pkl"


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record end time
        print(
            f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper
