"""
[EN]
This script inspects the structure of the preprocessed HDF5 dataset.
It lists all groups and datasets, including their shapes and data types,
to understand how event data is organized.

[GL]
Este script inspecciona a estrutura do dataset HDF5 preprocesado.
Lista todos os grupos e datasets, incluíndo as súas dimensións e tipos de datos,
para comprender como están organizados os datos de eventos.
"""

from pathlib import Path
import h5py

def print_group(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"[GROUP]   {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"[DATASET] {name} | shape={obj.shape} | dtype={obj.dtype}")

def main():
    repo_root = Path(__file__).resolve().parents[1]
    h5_path = repo_root / "tmp" / "sample_preprocessed.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    print(f"Opening: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        print("\n=== Top-level keys ===")
        for key in f.keys():
            print(f"- {key}")

        print("\n=== Full structure ===")
        f.visititems(print_group)

if __name__ == "__main__":
    main()