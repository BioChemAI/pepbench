import os
import random
import argparse
import subprocess
import pandas as pd

def run_cmd(cmd):
    print(f"[CMD] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def similar_split_dataset(data_path, threshold, output_dir = 'similar_splitter_data', keep_tmp = False, random_state=42):
    random.seed(random_state)
    print(f"[INFO] Using random seed = {random_state}")

    # Read data
    df = pd.read_csv(data_path)
    assert set(["id", "peps", "label"]).issubset(df.columns)

    # csv to FASTA
    fasta_path = "tmp_peptides.fasta"
    with open(fasta_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['id']}\n{row['peps']}\n")

    # Clustering
    print("\n[INFO] Start using MMseqs2 for clustering...")
    run_cmd("mmseqs createdb tmp_peptides.fasta peptidesDB")
    run_cmd(f"mmseqs cluster peptidesDB clusters tmp_mmseqs --min-seq-id {threshold}")
    run_cmd("mmseqs createtsv peptidesDB peptidesDB clusters clusters.tsv")

    # Read the clustering results
    clusters = pd.read_csv("clusters.tsv", sep="\t", header=None, names=["cluster", "id"])
    clusters["id"] = clusters["id"].astype(str)
    df["id"] = df["id"].astype(str)

    merged = df.merge(clusters, on="id", how="left")

    # Clustering division
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

    df_train = merged[merged["split"] == "train"][["id", "peps", "label"]]
    df_val = merged[merged["split"] == "val"][["id", "peps", "label"]]
    df_test = merged[merged["split"] == "test"][["id", "peps", "label"]]

    os.makedirs(output_dir, exist_ok=True)
    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"\n✅ Dataset division completed! The output file is located at：{output_dir}")
    print(f"train set：{len(df_train)} ")
    print(f"val set：{len(df_val)} ")
    print(f"test set：{len(df_test)} ")

    # Clean up temporary files
    if not keep_tmp:
        for f in ["tmp_peptides.fasta", "peptidesDB", "clusters", "tmp_mmseqs", "clusters.tsv"]:
            os.system(f"rm -rf {f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up temporary files")
    parser.add_argument("--data_path", type=str, required=True, help="Input CSV file (id, peps, label)")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold (default=0.8)")
    parser.add_argument("--output_dir", type=str, default="similar_splitter_data", help="Output directory")
    parser.add_argument("--keep_tmp", action="store_true", help="Whether to keep the temporary files or not")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    similar_split_dataset(data_path = args.data_path, threshold = args.threshold, output_dir = args.output_dir, random_state = args.random_state)
