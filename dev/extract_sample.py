"""
[EN]
This script extracts a reduced subset of the event dataset for fast debugging.
It copies one selected recording and a small set of ROIs into a new HDF5 file.

[GL]
Este script extrae un subconxunto reducido do dataset de eventos para depuración rápida.
Copia unha gravación seleccionada e un conxunto pequeno de ROI a un novo ficheiro HDF5.
"""

from pathlib import Path
import h5py

from dev.utils import get_repo_root, get_local_config, ensure_dir


def main():
    repo_root = get_repo_root()
    cfg = get_local_config()

    input_path = repo_root / cfg["data"]["h5_path"]
    output_path = repo_root / cfg["data"]["sample_h5_path"]

    ensure_dir(output_path.parent)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Escolla manual para depuración
    selected_recording = "22-01-05_17-00-00"
    selected_rois = ["N01", "N02"]

    print(f"[INFO] Input file:  {input_path}")
    print(f"[INFO] Output file: {output_path}")
    print(f"[INFO] Recording:   {selected_recording}")
    print(f"[INFO] ROIs:        {selected_rois}")

    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        if selected_recording not in f_in:
            raise KeyError(f"Recording not found: {selected_recording}")

        rec_in = f_in[selected_recording]
        rec_out = f_out.create_group(selected_recording)

        # copiar atributos do recording se existen
        for key, value in rec_in.attrs.items():
            rec_out.attrs[key] = value

        for roi in selected_rois:
            if roi not in rec_in:
                print(f"[WARN] ROI not found in recording: {roi}")
                continue

            print(f"[INFO] Copying {selected_recording}/{roi}")
            f_in.copy(f"{selected_recording}/{roi}", rec_out, name=roi)

    print("[DONE] Sample file created successfully.")


if __name__ == "__main__":
    main()