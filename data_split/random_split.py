import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def random_split_dataset(data_path, test_size=0.1, val_size=0.1, random_state=42):
    # Data loading
    df = pd.read_csv(data_path)
    print(f"[INFO] Original dataset samples: {len(df)}")

    # Get original filename without extension
    base_name = os.path.splitext(os.path.basename(data_path))[0]

    # Split
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )

    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio, random_state=random_state, shuffle=True
    )

    print(f"[INFO] Train samples: {len(train_df)}")
    print(f"[INFO] Val samples: {len(val_df)}")
    print(f"[INFO] Test samples: {len(test_df)}")

    # Saved directory
    save_dir = os.path.join(os.path.dirname(data_path), f"splitter{random_state}")
    os.makedirs(save_dir, exist_ok=True)

    # Save files
    train_df.to_csv(os.path.join(save_dir, f"{base_name}_train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, f"{base_name}_val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, f"{base_name}_test.csv"), index=False)

    print(f"[INFO] Data saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split regression dataset into train/val/test")
    parser.add_argument("--data_path", type=str, required=True, help="Original CSV data path")
    parser.add_argument("--random_state", type=int, default=111, help="Random seed")
    args = parser.parse_args()

    random_split_dataset(data_path=args.data_path, random_state=args.random_state)
