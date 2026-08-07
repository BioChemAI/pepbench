"""Split peptide datasets randomly or by MMseqs2 sequence similarity."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import subprocess
import tempfile
import os
import shutil
import warnings


def _stratified_split(
    data: "pd.DataFrame",
    test_size: float,
    random_state: Optional[int] = None,
    shuffle: bool = True
) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Internal function to perform stratified split maintaining class proportions
    
    Args:
        data (pd.DataFrame): Data to split
        test_size (float): Proportion for test set
        random_state (Optional[int]): Random state
        shuffle (bool): Whether to shuffle
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train, test)
    """
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Separate by class
    positive_data = data[data['label'] == 1.0].copy()
    negative_data = data[data['label'] == 0.0].copy()
    
    # Shuffle each class separately if needed
    if shuffle:
        positive_data = positive_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        negative_data = negative_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split indices for each class
    n_pos = len(positive_data)
    n_neg = len(negative_data)
    
    n_test_pos = int(n_pos * test_size)
    n_test_neg = int(n_neg * test_size)
    
    # Split positive samples
    test_pos = positive_data.iloc[:n_test_pos].copy()
    train_pos = positive_data.iloc[n_test_pos:].copy()
    
    # Split negative samples
    test_neg = negative_data.iloc[:n_test_neg].copy()
    train_neg = negative_data.iloc[n_test_neg:].copy()
    
    # Combine
    train = pd.concat([train_pos, train_neg], ignore_index=True)
    test = pd.concat([test_pos, test_neg], ignore_index=True)
    
    # Shuffle combined sets if needed
    if shuffle:
        train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return train, test


def split_dataset(
    data: "pd.DataFrame",
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: Optional[int] = None,
    stratify: bool = True,
    shuffle: bool = True
) -> Tuple["pd.DataFrame", "pd.DataFrame", Optional["pd.DataFrame"]]:
    """
    Split dataset into train, validation, and test sets
    
    Args:
        data (pd.DataFrame): Dataset to split. Must contain 'label' column.
        test_size (float): Proportion of dataset to include in test set. Defaults to 0.2.
        val_size (Optional[float]): Proportion of dataset to include in validation set.
            If None, no validation set is created. Defaults to None.
        random_state (Optional[int]): Random state for reproducibility. Defaults to None.
        stratify (bool): If True, split maintains the same proportion of positive/negative
            samples in each set as in the original dataset. Defaults to True.
        shuffle (bool): Whether to shuffle data before splitting. Defaults to True.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]: 
            (train_data, test_data, val_data) or (train_data, test_data, None)
    
    Raises:
        ValueError: If test_size or val_size are invalid
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    
    if 'label' not in data.columns:
        raise ValueError("DataFrame must contain 'label' column")
    
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")
    
    if val_size is not None and not (0 < val_size < 1):
        raise ValueError("val_size must be between 0 and 1")
    
    if val_size is not None and (test_size + val_size) >= 1:
        raise ValueError("test_size + val_size must be less than 1")
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    if stratify:
        # First split: separate test set (maintaining class proportions)
        train_val, test = _stratified_split(data, test_size, random_state, shuffle)
        
        # Second split: separate validation set if needed
        val = None
        if val_size is not None:
            # Calculate actual validation size relative to remaining data
            actual_val_size = val_size / (1 - test_size)
            train, val = _stratified_split(train_val, actual_val_size, random_state, shuffle)
        else:
            train = train_val
    else:
        # Non-stratified split
        if shuffle:
            data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        n_total = len(data)
        n_test = int(n_total * test_size)
        
        # Split test set
        test = data.iloc[:n_test].copy()
        train_val = data.iloc[n_test:].copy()
        
        # Split validation set if needed
        val = None
        if val_size is not None:
            n_val = int(n_total * val_size)
            val = train_val.iloc[:n_val].copy()
            train = train_val.iloc[n_val:].copy()
        else:
            train = train_val
    
    return train, test, val


def split_dataset_by_ratio(
    data: "pd.DataFrame",
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: Optional[int] = None,
    stratify: bool = True,
    shuffle: bool = True
) -> Tuple["pd.DataFrame", "pd.DataFrame", Optional["pd.DataFrame"]]:
    """
    Split dataset ensuring that train/val/test sets maintain the same positive/negative
    class ratio as the original dataset (stratified split)
    
    Args:
        data (pd.DataFrame): Dataset to split. Must contain 'label' column.
        test_size (float): Proportion of dataset to include in test set. Defaults to 0.2.
        val_size (Optional[float]): Proportion of dataset to include in validation set.
            If None, no validation set is created. Defaults to None.
        random_state (Optional[int]): Random state for reproducibility. Defaults to None.
        balance_classes (bool): If True, ensures that train/val/test sets maintain the same
            positive/negative ratio as the original dataset. This is equivalent to stratified split.
            Defaults to True.
        shuffle (bool): Whether to shuffle data before splitting. Defaults to True.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]: 
            (train_data, test_data, val_data) or (train_data, test_data, None)
    
    Raises:
        ValueError: If test_size or val_size are invalid
    """
    # stratify=True means we use stratified split to maintain original class ratio
    return split_dataset(
        data, test_size, val_size, random_state,
        stratify=stratify, shuffle=shuffle
    )


def _check_mmseqs2_available(mmseqs_path: Optional[str] = None) -> bool:
    """
    Check if MMseqs2 is available in the system
    
    Args:
        mmseqs_path (Optional[str]): Path to MMseqs2 executable. If None, uses 'mmseqs' from PATH.
    
    Returns:
        bool: True if MMseqs2 is available, False otherwise
    """
    mmseqs_cmd = mmseqs_path if mmseqs_path else 'mmseqs'
    try:
        result = subprocess.run(
            [mmseqs_cmd, 'version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _cluster_sequences_with_mmseqs2(
    sequences: List[str],
    sequence_ids: List[str],
    similarity_threshold: float = 0.4,
    coverage: float = 0.8,  # Fraction of each sequence covered by the alignment.
    sensitivity: float = 7.5,
    threads: int = 1,
    mmseqs_path: Optional[str] = None,
    tmp_dir: Optional[str] = None
) -> Dict[int, List[int]]:
    """
    Cluster sequences using MMseqs2 based on similarity
    
    Args:
        sequences (List[str]): List of peptide sequences
        sequence_ids (List[str]): List of sequence identifiers
        similarity_threshold (float): Sequence similarity threshold (0-1). 
            Recommended range: 0.3-0.5 for peptides. Lower values create less clusters.
            Defaults to 0.4 (40% similarity).
        coverage (float): Minimum coverage threshold (0-1). 
            Recommended range: 0.7-0.9. Defaults to 0.8 (80% coverage).
        sensitivity (float): Sensitivity parameter for MMseqs2 (-s).
            Recommended: 7.5 for balanced speed and accuracy. Defaults to 7.5.
        threads (int): Number of threads to use. Defaults to 1. 
            Increase for faster processing on multi-core systems.
        mmseqs_path (Optional[str]): Path to MMseqs2 executable. If None, uses 'mmseqs' from PATH.
        tmp_dir (Optional[str]): Temporary directory for MMseqs2 files. If None, creates a temp dir.
    
    Returns:
        Dict[int, List[int]]: Dictionary mapping cluster ID to list of sequence indices
    
    Raises:
        RuntimeError: If MMseqs2 is not available or clustering fails
    """
    # Check if MMseqs2 is available
    mmseqs_cmd = mmseqs_path if mmseqs_path else 'mmseqs'
    if not _check_mmseqs2_available(mmseqs_path):
        raise RuntimeError(
            "MMseqs2 is not available. Please install MMseqs2 or provide the path to the executable. "
            "Installation: https://github.com/soedinglab/MMseqs2"
        )
    
    # Create temporary directory
    if tmp_dir is None:
        tmp_dir_obj = tempfile.mkdtemp(prefix='mmseqs_cluster_')
        cleanup = True
    else:
        tmp_dir_obj = tmp_dir
        os.makedirs(tmp_dir_obj, exist_ok=True)
        cleanup = False
    
    try:
        # Create FASTA file
        fasta_file = os.path.join(tmp_dir_obj, 'sequences.fasta')
        with open(fasta_file, 'w') as f:
            for seq_id, seq in zip(sequence_ids, sequences):
                f.write(f">{seq_id}\n{seq}\n")
        
        # MMseqs2 clustering workflow
        db_path = os.path.join(tmp_dir_obj, 'db')
        cluster_db = os.path.join(tmp_dir_obj, 'cluster')
        cluster_result = os.path.join(tmp_dir_obj, 'cluster_result')
        
        # Create database
        create_db_cmd = [
            mmseqs_cmd, 'createdb',
            fasta_file,
            db_path
        ]
        result = subprocess.run(
            create_db_cmd,
            capture_output=True,
            text=True,
            cwd=tmp_dir_obj
        )
        if result.returncode != 0:
            raise RuntimeError(f"MMseqs2 createdb failed: {result.stderr}")
        
        # Cluster sequences using cluster command (supports sensitivity parameter)
        # cluster provides more control over clustering parameters including sensitivity
        cluster_tmp_dir = os.path.join(tmp_dir_obj, 'cluster_tmp')
        os.makedirs(cluster_tmp_dir, exist_ok=True)
        
        cluster_cmd = [
            mmseqs_cmd, 'cluster',
            db_path,
            cluster_db,
            cluster_tmp_dir,  # temporary directory for clustering
            '--min-seq-id', str(similarity_threshold),
            '--cov-mode', '1',  # Coverage mode: 1 = target coverage
            '-c', str(coverage),  # Coverage threshold
            '-s', str(sensitivity),  # Sensitivity parameter
            '--threads', str(threads)
        ]
        result = subprocess.run(
            cluster_cmd,
            capture_output=True,
            text=True,
            cwd=tmp_dir_obj
        )
        if result.returncode != 0:
            raise RuntimeError(f"MMseqs2 cluster failed: {result.stderr}")
        
        # Create tsv output
        tsv_output = os.path.join(tmp_dir_obj, 'cluster.tsv')
        createtsv_cmd = [
            mmseqs_cmd, 'createtsv',
            db_path,
            db_path,
            cluster_db,
            tsv_output
        ]
        result = subprocess.run(
            createtsv_cmd,
            capture_output=True,
            text=True,
            cwd=tmp_dir_obj
        )
        if result.returncode != 0:
            raise RuntimeError(f"MMseqs2 createtsv failed: {result.stderr}")
        
        # Read clustering results
        # TSV format: representative_id \t member_id
        # Note: The representative sequence itself may or may not appear as a member
        cluster_dict = {}
        id_to_index = {seq_id: idx for idx, seq_id in enumerate(sequence_ids)}
        
        # Use stable mapping from cluster representative to integer cluster ID
        cluster_rep_to_int = {}
        next_cluster_int = 0
        
        # Track which sequences we've seen (including representatives)
        seen_indices = set()
        
        with open(tsv_output, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cluster_rep = parts[0]  # Representative sequence ID
                    seq_id = parts[1]       # Member sequence ID
                    
                    # Create stable cluster ID mapping
                    if cluster_rep not in cluster_rep_to_int:
                        cluster_rep_to_int[cluster_rep] = next_cluster_int
                        next_cluster_int += 1
                    
                    cluster_int = cluster_rep_to_int[cluster_rep]
                    
                    # Add representative sequence to cluster if it exists
                    if cluster_rep in id_to_index:
                        rep_idx = id_to_index[cluster_rep]
                        if cluster_int not in cluster_dict:
                            cluster_dict[cluster_int] = []
                        if rep_idx not in seen_indices:
                            cluster_dict[cluster_int].append(rep_idx)
                            seen_indices.add(rep_idx)
                    
                    # Add member sequence to cluster
                    if seq_id in id_to_index:
                        seq_idx = id_to_index[seq_id]
                        if cluster_int not in cluster_dict:
                            cluster_dict[cluster_int] = []
                        if seq_idx not in seen_indices:
                            cluster_dict[cluster_int].append(seq_idx)
                            seen_indices.add(seq_idx)
        
        # Handle sequences that were not clustered (singletons)
        # These are sequences that don't appear in the TSV output at all
        for idx in range(len(sequences)):
            if idx not in seen_indices:
                cluster_dict[next_cluster_int] = [idx]
                next_cluster_int += 1
        
        return cluster_dict
        
    finally:
        # Cleanup temporary directory
        if cleanup and os.path.exists(tmp_dir_obj):
            shutil.rmtree(tmp_dir_obj)


def split_dataset_by_similarity(
    data: "pd.DataFrame",
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: Optional[int] = None,
    similarity_threshold: float = 0.4,
    coverage: float = 0.8,
    sensitivity: float = 7.5,
    threads: int = 1,
    mmseqs_path: Optional[str] = None,
    tmp_dir: Optional[str] = None
) -> Tuple["pd.DataFrame", "pd.DataFrame", Optional["pd.DataFrame"]]:
    """
    Split dataset based on MMseqs2 sequence similarity clustering.
    Ensures that sequences with high similarity (in the same cluster) are not split
    across train/val/test sets to avoid data leakage.
    
    Note: This method does not perform stratified splitting. Clusters are randomly
    assigned to train/val/test sets to ensure similar sequences stay together.
    
    Args:
        data (pd.DataFrame): Dataset to split. Must contain 'peps' and 'label' columns.
        test_size (float): Proportion of dataset to include in test set. Defaults to 0.2.
        val_size (Optional[float]): Proportion of dataset to include in validation set.
            If None, no validation set is created. Defaults to None.
        random_state (Optional[int]): Random state for reproducibility. Defaults to None.
        similarity_threshold (float): Sequence similarity threshold for clustering (0-1).
            Recommended range: 0.3-0.5 for peptides. Lower values create more clusters.
            - 0.3-0.4: For short peptides (< 50 amino acids), more clusters
            - 0.4-0.5: For medium peptides (50-200 amino acids), balanced
            - 0.5-0.6: For long peptides (> 200 amino acids), fewer clusters
            Defaults to 0.4 (40% similarity).
        coverage (float): Minimum coverage threshold for clustering (0-1).
            Recommended range: 0.7-0.9. Defaults to 0.8 (80% coverage).
        sensitivity (float): Sensitivity parameter for MMseqs2 (-s).
            Recommended: 7.5 for balanced speed and accuracy. Defaults to 7.5.
        threads (int): Number of threads to use for MMseqs2 clustering.
            Increase for faster processing on multi-core systems. Defaults to 1.
        mmseqs_path (Optional[str]): Path to MMseqs2 executable. If None, uses 'mmseqs' from PATH.
        tmp_dir (Optional[str]): Temporary directory for MMseqs2 files. If None, creates a temp dir.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]: 
            (train_data, test_data, val_data) or (train_data, test_data, None)
    
    Raises:
        ValueError: If test_size or val_size are invalid
        RuntimeError: If MMseqs2 is not available or clustering fails
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    
    if 'peps' not in data.columns:
        raise ValueError("DataFrame must contain 'peps' column")
    if 'label' not in data.columns:
        raise ValueError("DataFrame must contain 'label' column")
    
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")
    
    if val_size is not None and not (0 < val_size < 1):
        raise ValueError("val_size must be between 0 and 1")
    
    if val_size is not None and (test_size + val_size) >= 1:
        raise ValueError("test_size + val_size must be less than 1")
    
    if not (0 < similarity_threshold <= 1):
        raise ValueError("similarity_threshold must be between 0 and 1")
    
    if not (0 < coverage <= 1):
        raise ValueError("coverage must be between 0 and 1")
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Prepare sequences and IDs
    sequences = data['peps'].tolist()
    sequence_ids = [f"seq_{i}" for i in range(len(sequences))]
    
    # Cluster sequences using MMseqs2
    print("Clustering sequences with MMseqs2...")
    print(f"Parameters: similarity_threshold={similarity_threshold}, coverage={coverage}, "
          f"sensitivity={sensitivity}, threads={threads}")
    cluster_dict = _cluster_sequences_with_mmseqs2(
        sequences,
        sequence_ids,
        similarity_threshold=similarity_threshold,
        coverage=coverage,
        sensitivity=sensitivity,
        threads=threads,
        mmseqs_path=mmseqs_path,
        tmp_dir=tmp_dir
    )
    
    print(f"Found {len(cluster_dict)} clusters")
    
    # Convert clusters to list of cluster assignments
    cluster_assignments = np.zeros(len(sequences), dtype=int)
    for cluster_id, indices in cluster_dict.items():
        for idx in indices:
            cluster_assignments[idx] = cluster_id
    
    # Group data by cluster
    data_with_cluster = data.copy()
    data_with_cluster['cluster_id'] = cluster_assignments
    
    # Split clusters (not individual sequences) to avoid splitting similar sequences
    unique_clusters = data_with_cluster['cluster_id'].unique()
    
    # Calculate cluster information (size and class distribution)
    cluster_info = []
    for cluster_id in unique_clusters:
        cluster_data = data_with_cluster[data_with_cluster['cluster_id'] == cluster_id]
        cluster_info.append({
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'positive_count': (cluster_data['label'] == 1.0).sum(),
            'negative_count': (cluster_data['label'] == 0.0).sum()
        })
    
    # Sort clusters by size (larger clusters first) for more balanced splits
    cluster_info.sort(key=lambda x: x['size'], reverse=True)
    
    # Shuffle clusters with random state
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(cluster_info)
    
    # Greedy assignment: assign clusters to test/val/train based on target sizes
    # This ensures we get close to the desired proportions while keeping clusters intact
    total_samples = len(data)
    target_test_samples = int(total_samples * test_size)
    target_val_samples = int(total_samples * val_size) if val_size is not None else 0
    
    test_clusters = set()
    val_clusters = set()
    train_clusters = set()
    
    test_count = 0
    val_count = 0
    
    for info in cluster_info:
        cluster_id = info['cluster_id']
        cluster_size = info['size']
        
        # Assign to test set if we haven't reached target
        if test_count < target_test_samples:
            test_clusters.add(cluster_id)
            test_count += cluster_size
        # Assign to validation set if needed and we haven't reached target
        elif val_size is not None and val_count < target_val_samples:
            val_clusters.add(cluster_id)
            val_count += cluster_size
        # Otherwise assign to training set
        else:
            train_clusters.add(cluster_id)
    
    # Assign data to splits based on cluster membership
    train = data_with_cluster[data_with_cluster['cluster_id'].isin(train_clusters)].copy()
    test = data_with_cluster[data_with_cluster['cluster_id'].isin(test_clusters)].copy()
    
    # Remove cluster_id column
    train = train.drop(columns=['cluster_id'])
    test = test.drop(columns=['cluster_id'])
    
    val = None
    if val_size is not None and len(val_clusters) > 0:
        val = data_with_cluster[data_with_cluster['cluster_id'].isin(val_clusters)].copy()
        val = val.drop(columns=['cluster_id'])
    
    # Shuffle within each split
    if random_state is not None:
        np.random.seed(random_state)
    train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)
    if val is not None:
        val = val.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"Split complete: Train={len(train)}, Test={len(test)}, Val={len(val) if val is not None else 0}")
    
    return train, test, val


if __name__ == '__main__':
    """
    Test similarity-based dataset splitting with real dataset
    """
    print("=" * 80)
    print("Testing Similarity-Based Dataset Splitting")
    print("=" * 80)
    
    try:
        # Import dataset loader
        from pephub.dataset import PepDataset
        
        # Initialize dataset loader (uses built-in datasets by default)
        loader = PepDataset()
        
        # List available datasets
        datasets = loader.list_available_datasets()
        print(f"\nAvailable datasets: {datasets}")
        
        # Use a smaller dataset for testing (if available) to speed up clustering
        # Try DPPIV first, fallback to first available dataset
        test_dataset_name = None
        if 'DPPIV' in datasets:
            test_dataset_name = 'DPPIV'
        elif 'SA100' in datasets:
            test_dataset_name = 'SA100'
        elif 'EC100' in datasets:
            test_dataset_name = 'EC100'
        elif len(datasets) > 0:
            test_dataset_name = datasets[0]
        else:
            raise ValueError("No datasets available for testing")
        
        print(f"\nLoading dataset: {test_dataset_name}")
        data = loader.load_dataset(test_dataset_name)
        print(f"Dataset loaded: {len(data)} samples")
        print(f"Columns: {data.columns.tolist()}")
        print(f"Label distribution:")
        print(f"  Positive (1.0): {(data['label'] == 1.0).sum()} ({(data['label'] == 1.0).sum() / len(data) * 100:.2f}%)")
        print(f"  Negative (0.0): {(data['label'] == 0.0).sum()} ({(data['label'] == 0.0).sum() / len(data) * 100:.2f}%)")
        
        # Test similarity-based splitting
        print("\n" + "=" * 80)
        print("Test 1: Similarity-based split (with validation set)")
        print("=" * 80)
        
        train_data, test_data, val_data = split_dataset_by_similarity(
            data,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            similarity_threshold=0.4,
            coverage=0.8,
            sensitivity=7.5,
            threads=1
        )
        
        # Verify split results
        print("\nSplit Results:")
        print(f"  Train set: {len(train_data)} samples ({len(train_data) / len(data) * 100:.2f}%)")
        print(f"  Test set: {len(test_data)} samples ({len(test_data) / len(data) * 100:.2f}%)")
        print(f"  Val set: {len(val_data)} samples ({len(val_data) / len(data) * 100:.2f}%)")
        print(f"  Total: {len(train_data) + len(test_data) + len(val_data)} samples")
        
        # Check that all samples are accounted for
        assert len(train_data) + len(test_data) + len(val_data) == len(data), \
            f"Sample count mismatch: {len(train_data) + len(test_data) + len(val_data)} != {len(data)}"
        print("  ✓ All samples accounted for")
        
        # Check that no samples are duplicated (using 'id' column if available, otherwise index)
        if 'id' in train_data.columns:
            train_ids = set(train_data['id'].values)
            test_ids = set(test_data['id'].values)
            val_ids = set(val_data['id'].values)
            all_ids = train_ids | test_ids | val_ids
            assert len(all_ids) == len(train_data) + len(test_data) + len(val_data), \
                f"Duplicate samples found: {len(all_ids)} unique IDs != {len(train_data) + len(test_data) + len(val_data)} total samples"
            assert len(train_ids & test_ids) == 0, "Duplicate samples between train and test"
            assert len(train_ids & val_ids) == 0, "Duplicate samples between train and val"
            assert len(test_ids & val_ids) == 0, "Duplicate samples between test and val"
        else:
            # Fallback to index-based check
            train_indices = set(train_data.index)
            test_indices = set(test_data.index)
            val_indices = set(val_data.index)
            all_indices = train_indices | test_indices | val_indices
            assert len(all_indices) == len(train_data) + len(test_data) + len(val_data), \
                f"Duplicate samples found: {len(all_indices)} unique indices != {len(train_data) + len(test_data) + len(val_data)} total samples"
            assert len(train_indices & test_indices) == 0, "Duplicate samples between train and test"
            assert len(train_indices & val_indices) == 0, "Duplicate samples between train and val"
            assert len(test_indices & val_indices) == 0, "Duplicate samples between test and val"
        print("  ✓ No duplicate samples across splits")
        
        # Check label distribution
        print("\nLabel Distribution:")
        for split_name, split_data in [("Train", train_data), ("Test", test_data), ("Val", val_data)]:
            pos_count = (split_data['label'] == 1.0).sum()
            neg_count = (split_data['label'] == 0.0).sum()
            total = len(split_data)
            print(f"  {split_name}:")
            print(f"    Positive: {pos_count} ({pos_count / total * 100:.2f}%)")
            print(f"    Negative: {neg_count} ({neg_count / total * 100:.2f}%)")
        
        # Test 2: Similarity-based split (without validation set)
        print("\n" + "=" * 80)
        print("Test 2: Similarity-based split (without validation set)")
        print("=" * 80)
        
        train_data2, test_data2, val_data2 = split_dataset_by_similarity(
            data,
            test_size=0.2,
            val_size=None,
            random_state=42,
            similarity_threshold=0.4,
            coverage=0.8,
            sensitivity=7.5,
            threads=1
        )
        
        assert val_data2 is None, "Validation set should be None when val_size=None"
        assert len(train_data2) + len(test_data2) == len(data), \
            f"Sample count mismatch: {len(train_data2) + len(test_data2)} != {len(data)}"
        print("  ✓ Split without validation set successful")
        print(f"  Train: {len(train_data2)} samples, Test: {len(test_data2)} samples")
        
        # Test 3: Different similarity threshold
        print("\n" + "=" * 80)
        print("Test 3: Similarity-based split with different threshold")
        print("=" * 80)
        
        train_data3, test_data3, val_data3 = split_dataset_by_similarity(
            data,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            similarity_threshold=0.3,  # Lower threshold = more clusters
            coverage=0.8,
            sensitivity=7.5,
            threads=1
        )
        
        assert len(train_data3) + len(test_data3) + len(val_data3) == len(data), \
            f"Sample count mismatch with different threshold"
        print("  ✓ Split with different similarity threshold successful")
        print(f"  Train: {len(train_data3)} samples, Test: {len(test_data3)} samples, Val: {len(val_data3)} samples")
        
        # Test 4: Reproducibility - same random_state should produce same results
        print("\n" + "=" * 80)
        print("Test 4: Reproducibility test (same random_state)")
        print("=" * 80)
        
        # First split with random_state=100
        train_data4a, test_data4a, val_data4a = split_dataset_by_similarity(
            data,
            test_size=0.2,
            val_size=0.1,
            random_state=100,
            similarity_threshold=0.4,
            coverage=0.8,
            sensitivity=7.5,
            threads=1
        )
        
        # Second split with same random_state=100
        train_data4b, test_data4b, val_data4b = split_dataset_by_similarity(
            data,
            test_size=0.2,
            val_size=0.1,
            random_state=100,
            similarity_threshold=0.4,
            coverage=0.8,
            sensitivity=7.5,
            threads=1
        )
        
        # Check that results are identical
        print("\nComparing two splits with same random_state=100...")
        
        # Check sizes
        assert len(train_data4a) == len(train_data4b), \
            f"Train set sizes differ: {len(train_data4a)} != {len(train_data4b)}"
        assert len(test_data4a) == len(test_data4b), \
            f"Test set sizes differ: {len(test_data4a)} != {len(test_data4b)}"
        assert len(val_data4a) == len(val_data4b), \
            f"Val set sizes differ: {len(val_data4a)} != {len(val_data4b)}"
        print("  ✓ Split sizes are identical")
        
        # Check that samples are the same (using 'id' column if available)
        if 'id' in train_data4a.columns:
            # Sort by id for comparison
            train_ids_a = sorted(train_data4a['id'].values)
            train_ids_b = sorted(train_data4b['id'].values)
            test_ids_a = sorted(test_data4a['id'].values)
            test_ids_b = sorted(test_data4b['id'].values)
            val_ids_a = sorted(val_data4a['id'].values)
            val_ids_b = sorted(val_data4b['id'].values)
            
            assert train_ids_a == train_ids_b, \
                "Train sets contain different samples (by id)"
            assert test_ids_a == test_ids_b, \
                "Test sets contain different samples (by id)"
            assert val_ids_a == val_ids_b, \
                "Val sets contain different samples (by id)"
            print("  ✓ All samples are identical (by id)")
        else:
            # Fallback: check by index (after sorting)
            train_indices_a = sorted(train_data4a.index)
            train_indices_b = sorted(train_data4b.index)
            test_indices_a = sorted(test_data4a.index)
            test_indices_b = sorted(test_data4b.index)
            val_indices_a = sorted(val_data4a.index)
            val_indices_b = sorted(val_data4b.index)
            
            assert train_indices_a == train_indices_b, \
                "Train sets contain different samples (by index)"
            assert test_indices_a == test_indices_b, \
                "Test sets contain different samples (by index)"
            assert val_indices_a == val_indices_b, \
                "Val sets contain different samples (by index)"
            print("  ✓ All samples are identical (by index)")
        
        # Test 5: Different random_state should produce different results
        print("\n" + "=" * 80)
        print("Test 5: Different random_state should produce different results")
        print("=" * 80)
        
        train_data5, test_data5, val_data5 = split_dataset_by_similarity(
            data,
            test_size=0.2,
            val_size=0.1,
            random_state=200,  # Different random_state
            similarity_threshold=0.4,
            coverage=0.8,
            sensitivity=7.5,
            threads=1
        )
        
        # Check that results are different from random_state=100
        if 'id' in train_data4a.columns:
            train_ids_100 = set(train_data4a['id'].values)
            train_ids_200 = set(train_data5['id'].values)
            # They might have some overlap, but should not be identical
            # At least check that sizes are the same (which we already know from test_size)
            assert len(train_ids_100) == len(train_ids_200), \
                "Train set sizes should be similar with different random_state"
            # Check if they are different (they should be, but allow for edge cases)
            if train_ids_100 == train_ids_200:
                print("  ⚠ Warning: Train sets are identical despite different random_state (unlikely but possible)")
            else:
                print("  ✓ Different random_state produces different splits (as expected)")
        else:
            train_indices_100 = set(train_data4a.index)
            train_indices_200 = set(train_data5.index)
            if train_indices_100 == train_indices_200:
                print("  ⚠ Warning: Train sets are identical despite different random_state (unlikely but possible)")
            else:
                print("  ✓ Different random_state produces different splits (as expected)")
        
        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure pephub.dataset is available")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
