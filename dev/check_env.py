'''
[EN]
This script checks the environment for the required dependencies and file structure for the project. 
It verifies the presence of numpy, h5py, and torch, and checks for the existence of the preprocessed.h5 file and the src/ directory.

[GL]
Este ficheiro verifica o entorno para as dependecias e a estrutura necesaria para o proxecto. 
Verifica a presenza de numpy, h5py e torch, e comproba a existencia do ficheiro preprocessed.h5 e do directorio src/.
'''
from pathlib import Path

def main():
    print("=== Environment check ===")

    try:
        import numpy as np
        print(f"[OK] numpy {np.__version__}")
    except Exception as e:
        print(f"[FAIL] numpy: {e}")

    try:
        import h5py
        print(f"[OK] h5py {h5py.__version__}")
    except Exception as e:
        print(f"[FAIL] h5py: {e}")

    try:
        import torch
        print(f"[OK] torch {torch.__version__}")
    except Exception as e:
        print(f"[FAIL] torch: {e}")

    repo_root = Path(__file__).resolve().parents[1]
    print(f"[INFO] repo root: {repo_root}")

    h5_path = repo_root / "data" / "preprocessed.h5"
    print(f"[INFO] expected H5 path: {h5_path}")

    if h5_path.exists():
        print("[OK] preprocessed.h5 found")
    else:
        print("[WARN] preprocessed.h5 not found")

    src_path = repo_root / "src"
    if src_path.exists():
        print("[OK] src/ found")
    else:
        print("[FAIL] src/ not found")

    print("=== Done ===")

if __name__ == "__main__":
    main()