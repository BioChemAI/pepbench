# No-extra-files version: similarity-based dataset splitting with MMseqs2
import os
import argparse
import pandas as pd
import subprocess
import random


def run_cmd(cmd):
    """Execute a shell command with logging."""
    print(f"[CMD] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def clean_tmp():
    """Remove and recreate the temporary MMseqs2 directory."""
    os.system("rm -rf tmp_mmseqs")
    os.makedirs("tmp_mmseqs", exist_ok=True)


def similar_split_dataset(args):
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        print(f"[INFO] Random seed set to: {args.seed}")
    else:
        print("[INFO] Random seed not set; results may vary between runs.")

    # Load input CSV file
    df = pd.read_csv(args.input)
    assert set(["id", "peps", "label"]).issubset(df.columns), \
        "Input CSV must contain the columns: id, peps, label"
    print(f"[INFO] Loaded {len(df)} sequences")

    data_name = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    # Write sequences to FASTA format (sanitize invalid characters)
    fasta_path = f"{data_name}.fasta"
    with open(fasta_path, "w") as f:
        for _, row in df.iterrows():
            seq = str(row["peps"]).strip().upper()
            seq = "".join([c for c in seq if c.isalpha()])
            f.write(f">{row['id']}\n{seq}\n")

    # Clean temporary directory to avoid conflicts
    clean_tmp()

    # Run MMseqs2 clustering with stability-oriented parameters
    print("\n[INFO] Running MMseqs2 clustering...")

    mmseqs_cmd = (
        f"mmseqs easy-cluster {fasta_path} {data_name}_cluster tmp_mmseqs "
        f"--min-seq-id {args.threshold} "
        f"--cov-mode 0 -c 0.8 "
        f"--threads {args.threads} "
        f"--db-load-mode 0 "
        f"--split-memory-limit 4G "
        f"-v 3"
    )

    try:
        run_cmd(mmseqs_cmd)
    except subprocess.CalledProcessError:
        print(
            "\n[ERROR] MMseqs2 failed. Retrying with a safer configuration "
            "(reduced number of threads)...\n"
        )
        fallback_cmd = mmseqs_cmd.replace(
            f"--threads {args.threads}", "--threads 2"
        )
        run_cmd(fallback_cmd)

    # Load clustering results
    cluster_tsv = f"{data_name}_cluster_cluster.tsv"
    clusters = pd.read_csv(
        cluster_tsv, sep="\t", header=None, names=["cluster", "id"]
    )
    clusters["id"] = clusters["id"].astype(str)
    df["id"] = df["id"].astype(str)

    merged = df.merge(clusters, on="id", how="left")

    # Split clusters into training, validation, and test sets
    cluster_ids = merged["cluster"].dropna().unique().tolist()
    random.shuffle(cluster_ids)

    n = len(cluster_ids)
    train_cut = int(0.8 * n)
    val_cut = int(0.9 * n)

    train_clusters = set(cluster_ids[:train_cut])
    val_clusters = set(cluster_ids[train_cut:val_cut])
    test_clusters = set(cluster_ids[val_cut:])

    def assign_split(row):
        if row["cluster"] in train_clusters:
            return "train"
        elif row["cluster"] in val_clusters:
            return "val"
        else:
            return "test"

    merged["split"] = merged.apply(assign_split, axis=1)

    # Save split datasets
    df_train = merged[merged["split"] == "train"][["id", "peps", "label"]]
    df_val = merged[merged["split"] == "val"][["id", "peps", "label"]]
    df_test = merged[merged["split"] == "test"][["id", "peps", "label"]]

    df_train.to_csv(
        os.path.join(args.output_dir, f"{data_name}_train.csv"), index=False
    )
    df_val.to_csv(
        os.path.join(args.output_dir, f"{data_name}_val.csv"), index=False
    )
    df_test.to_csv(
        os.path.join(args.output_dir, f"{data_name}_test.csv"), index=False
    )

    print(f"\nDataset splitting completed. Files saved to: {args.output_dir}")
    print(
        f"Train: {len(df_train)} | "
        f"Validation: {len(df_val)} | "
        f"Test: {len(df_test)}"
    )

    # Remove temporary files if requested
    if not args.keep_tmp:
        clean_tmp()
        os.system(f"rm -rf {fasta_path} {data_name}_cluster*")
        print("[INFO] Temporary files removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Similarity-based dataset splitting using MMseqs2. "
            "Sequences are clustered by sequence identity, and clusters "
            "are assigned to training, validation, and test sets."
        )
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input CSV file containing id, peps, and label columns"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.8,
        help="Sequence identity threshold for MMseqs2 clustering"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="similar_splitter_data/splitter333",
        help="Output directory for the split CSV files"
    )
    parser.add_argument(
        "--keep_tmp", action="store_true",
        help="Keep intermediate MMseqs2 files"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible cluster splitting"
    )
    parser.add_argument(
        "--threads", type=int, default=8,
        help="Number of CPU threads used by MMseqs2"
    )

    args = parser.parse_args()
    similar_split_dataset(args)
