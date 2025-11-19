import os
import argparse
import pandas as pd
import subprocess
import random

def run_cmd(cmd):
    print(f"[CMD] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def clean_tmp():
    os.system("rm -rf tmp_mmseqs")
    os.makedirs("tmp_mmseqs", exist_ok=True)

def similar_split_dataset(args):
    # 0️⃣ Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        print(f"[INFO] Random seed set to: {args.seed}")
    else:
        print("[INFO] Random seed is not set, results may vary.")

    # 1️⃣ Read input CSV
    df = pd.read_csv(args.input)
    assert set(["id", "peps", "label"]).issubset(df.columns), \
        "Input file must contain columns: id, peps, label"
    print(f"[INFO] Loaded {len(df)} sequences")

    data_name = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    # 2️⃣ Write FASTA (clean illegal characters)
    fasta_path = f"{data_name}.fasta"
    with open(fasta_path, "w") as f:
        for _, row in df.iterrows():
            seq = str(row['peps']).strip().upper()
            seq = ''.join([c for c in seq if c.isalpha()])  # remove illegal chars
            f.write(f">{row['id']}\n{seq}\n")

    # 3️⃣ Clean old tmp directory
    clean_tmp()

    # 4️⃣ Run MMseqs2
    print("\n[INFO] Running MMseqs2 clustering...")

    mmseqs_cmd = (
        f"mmseqs easy-cluster {fasta_path} {data_name}_cluster tmp_mmseqs "
        f"--min-seq-id {args.threshold} "
        f"--cov-mode 0 -c 0.8 "
        f"--threads {args.threads} "
        f"--db-load-mode 0 "                 # prevent memory overuse
        f"--split-memory-limit 4G "          # improve stability
        f"-v 3"
    )

    try:
        run_cmd(mmseqs_cmd)
    except subprocess.CalledProcessError:
        print("\n[ERROR] MMseqs2 crashed. Retrying with reduced threads...\n")
        
        # fallback: safer but slower
        fallback_cmd = mmseqs_cmd.replace(f"--threads {args.threads}", "--threads 2")
        run_cmd(fallback_cmd)

    # 5️⃣ Read clustering results
    cluster_tsv = f"{data_name}_cluster_cluster.tsv"
    clusters = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["cluster", "id"])
    clusters["id"] = clusters["id"].astype(str)
    df["id"] = df["id"].astype(str)

    merged = df.merge(clusters, on="id", how="left")

    # 6️⃣ Cluster-based split
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

    # 7️⃣ Save results
    df_train = merged[merged["split"] == "train"][["id", "peps", "label"]]
    df_val = merged[merged["split"] == "val"][["id", "peps", "label"]]
    df_test = merged[merged["split"] == "test"][["id", "peps", "label"]]

    df_train.to_csv(os.path.join(args.output_dir, f"{data_name}_train.csv"), index=False)
    df_val.to_csv(os.path.join(args.output_dir, f"{data_name}_val.csv"), index=False)
    df_test.to_csv(os.path.join(args.output_dir, f"{data_name}_test.csv"), index=False)

    print(f"\n✅ Dataset splitting completed. Files saved to: {args.output_dir}")
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # 8️⃣ Clean temporary files
    if not args.keep_tmp:
        clean_tmp()
        os.system(f"rm -rf {fasta_path} {data_name}_cluster*")
        print("[INFO] Temporary files removed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Similarity-based dataset split using MMseqs2 (output CSV)")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--threshold", type=float, default=0.8, help="Minimum sequence identity threshold")
    parser.add_argument("--output_dir", type=str, default="similar_splitter_data", help="Output directory")
    parser.add_argument("--keep_tmp", action="store_true", help="Keep intermediate files")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads for MMseqs2")
    args = parser.parse_args()
    similar_split_dataset(args)
