"""
[EN]
Run reTAG proposal generation on the full dataset (test split).

[GL]
Executa a xeración de propostas reTAG sobre todo o dataset (split test).
"""

from dev.run_retag_debug import ProposalGenerator

def main():
    generator = ProposalGenerator(
        data_path="data/preprocessed.h5",  # ← CAMBIAR
        bin_width=0.01,
        percentile=5,
        nms_threshold=0.95,
    )

    df = generator.run()

    print("Done.")
    print(f"Total proposals: {len(df)}")

    # opcional: gardar resultados
    df.to_csv("tmp/retag_adaptive_full.csv", index=False)


if __name__ == "__main__":
    main()