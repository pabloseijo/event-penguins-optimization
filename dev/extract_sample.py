"""Extract a small HDF5 subset for fast local debugging.

Copies one recording and a fixed set of ROIs from the full dataset into a
lightweight sample file. Run from the event_penguins/ directory:

    python dev/extract_sample.py
"""

from pathlib import Path
import h5py

_RECORDING     = "22-01-05_17-00-00"
_ROIS          = ["N01", "N02"]
_INPUT_PATH    = Path("data/preprocessed.h5")
_OUTPUT_PATH   = Path("tmp/sample_preprocessed.h5")


def main() -> None:
    if not _INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {_INPUT_PATH}")

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input:     {_INPUT_PATH}")
    print(f"[INFO] Output:    {_OUTPUT_PATH}")
    print(f"[INFO] Recording: {_RECORDING}")
    print(f"[INFO] ROIs:      {_ROIS}")

    with h5py.File(_INPUT_PATH, "r") as f_in, h5py.File(_OUTPUT_PATH, "w") as f_out:
        if _RECORDING not in f_in:
            raise KeyError(f"Recording not found in dataset: {_RECORDING}")

        rec_in  = f_in[_RECORDING]
        rec_out = f_out.create_group(_RECORDING)

        for key, value in rec_in.attrs.items():
            rec_out.attrs[key] = value

        for roi in _ROIS:
            if roi not in rec_in:
                print(f"[WARN] ROI not found: {roi}")
                continue
            print(f"[INFO] Copying {_RECORDING}/{roi}")
            f_in.copy(f"{_RECORDING}/{roi}", rec_out, name=roi)

    print(f"[DONE] Sample saved to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
