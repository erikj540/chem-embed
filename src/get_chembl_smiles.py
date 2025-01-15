from pathlib import Path

import chembl_downloader
import click

from src.constants import DATA_DIR
from src.utils import get_chembl_smiles_file_name, pickle_and_compress


@click.command(
    help="""
    This script downloads the SMILES from ChEMBL (https://www.ebi.ac.uk/chembl/).\n
    The SMILES are pickled and saved to a file named chembl_v<ChEMBL version>_smiles.pkl \n
    """,
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.option(
    "--chembl_version",
    "-v",
    type=str,
    default="23",
    help="The ChEMBL version to use. Defaults to '23'.",
)
@click.option(
    "--out_dir",
    "-o",
    type=str,
    default=str(DATA_DIR),
    help=f"Directory where to save the output file. Defaults to {DATA_DIR}",
)
def get_chembl_smiles(chembl_version: str, out_dir: str):
    out_path = Path(out_dir) / get_chembl_smiles_file_name(
        chembl_version=chembl_version
    )
    if not out_path.exists():
        smiles = set(
            smiles
            for smiles in chembl_downloader.iterate_smiles(version=chembl_version)
        )
        pickle_and_compress(obj=smiles, file_path=out_path)


if __name__ == "__main__":
    get_chembl_smiles()
