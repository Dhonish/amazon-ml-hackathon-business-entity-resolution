import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, fbeta_score

try:
    from .utils import normalize_name, normalize_address, normalize_text, ProgressTracker
    from .features import make_features
    from .blocking import get_blocks
except ImportError:
    from utils import normalize_name, normalize_address, normalize_text, ProgressTracker
    from features import make_features
    from blocking import get_blocks


S1_SAMPLE_SIZE = 40000
CHUNK_SIZE = 100000
NEGATIVES_PER_S1 = 6
MAX_TARGET_LINES_MINING = 600000

RAW_COLUMNS = ["business_name", "business_address", "country"]


def fill_missing(df):
    df[RAW_COLUMNS] = df[RAW_COLUMNS].fillna("")
    return df


def load_data():
    print("[1/6] Loading training data and ground truth...", flush=True)
    t0 = time.time()
    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", encoding="latin1", dtype=str)
    s1 = fill_missing(s1)

    gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", encoding="latin1", dtype=str)
    print(f"Loaded {len(s1):,} S1 rows and {len(gt):,} Ground Truth rows in {time.time() - t0:.1f}s", flush=True)
    return s1, gt


def build_ground_truth(gt):
    truth = {}
    for row in gt.itertuples(index=False):
        s1_id = row.source1_entity_id
        value = str(row.matched_entity_ids) if not pd.isna(row.matched_entity_ids) else ""
        if value.strip():
            truth[s1_id] = set(x.strip() for x in value.split(",") if x.strip())
        else:
            truth[s1_id] = set()
    return truth


def sample_source1(s1, truth):
    matched = [eid for eid in s1["entity_id"] if truth.get(eid, set())]
    unmatched = [eid for eid in s1["entity_id"] if not truth.get(eid, set())]

    rng = np.random.default_rng(42)
    n_matched = min(S1_SAMPLE_SIZE // 2, len(matched))
    n_unmatched = min(S1_SAMPLE_SIZE - n_matched, len(unmatched))

    selected = set(rng.choice(matched, size=n_matched, replace=False))
    selected.update(rng.choice(unmatched, size=n_unmatched, replace=False))

    s1_sampled = s1[s1["entity_id"].isin(selected)].copy()
    print(f"[2/6] Sampled {len(s1_sampled):,} S1 entities ({n_matched:,} matched, {n_unmatched:,} unmatched)", flush=True)
    return s1_sampled


def build_block_index(s1):
    print("[3/6] Building S1 block index...", flush=True)
    block_to_s1 = {}
    tracker = ProgressTracker(len(s1), "S1 Blocking", update_interval_sec=5)

    for row in s1.itertuples(index=False):
        keys = get_blocks(row.business_name, row.business_address, row.country)
        for key in keys:
            block_to_s1.setdefault(key, []).append(row.entity_id)
        tracker.update(1)

    print(f"Built {len(block_to_s1):,} unique block keys.", flush=True)
    return block_to_s1


def find_candidate_ids(s1, truth, block_to_s1):
    print("[4/6] Mining positive and negative training candidates...", flush=True)
    negative_ids = {eid: [] for eid in s1["entity_id"]}
    positive_ids = set()
    for eid in s1["entity_id"]:
        positive_ids.update(truth.get(eid, set()))

    files = ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]
    t0 = time.time()
    total_lines = 0

    for file_path in files:
        if not os.path.exists(file_path) or total_lines >= MAX_TARGET_LINES_MINING:
            break
        print(f"  Scanning target file: {file_path}", flush=True)

        for chunk in pd.read_csv(file_path, sep="\t", encoding="latin1", dtype=str, chunksize=CHUNK_SIZE):
            chunk = fill_missing(chunk)
            for row in chunk.itertuples(index=False):
                tid = row.entity_id
                keys = get_blocks(row.business_name, row.business_address, row.country)

                matched_s1 = set()
                for key in keys:
                    if key in block_to_s1:
                        matched_s1.update(block_to_s1[key])

                for s1_id in matched_s1:
                    if tid in truth.get(s1_id, set()):
                        continue
                    current = negative_ids[s1_id]
                    if len(current) < NEGATIVES_PER_S1:
                        current.append(tid)

            total_lines += len(chunk)
            elapsed = time.time() - t0
            speed = total_lines / elapsed if elapsed > 0 else 0
            print(f"    [Candidate Mining] Processed {total_lines:,} lines | Speed: {speed:,.0f} lines/s | Elapsed: {elapsed:.0f}s", flush=True)
            if total_lines >= MAX_TARGET_LINES_MINING:
                print("  Reached target line limit for candidate mining.", flush=True)
                break

    return negative_ids, positive_ids


def load_required_targets(positive_ids, negative_ids):
    required_ids = set(positive_ids)
    for ids in negative_ids.values():
        required_ids.update(ids)

    print(f"[5/6] Loading {len(required_ids):,} required target entity records...", flush=True)
    records = {}
    files = ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]
    t0 = time.time()

    for file_path in files:
        if len(records) == len(required_ids) or not os.path.exists(file_path):
            break
        for chunk in pd.read_csv(file_path, sep="\t", encoding="latin1", dtype=str, chunksize=CHUNK_SIZE):
            chunk = fill_missing(chunk)
            selected = chunk[chunk["entity_id"].isin(required_ids)]
            for row in selected.itertuples(index=False):
                records[row.entity_id] = row
            if len(records) == len(required_ids):
                break

    print(f"Loaded {len(records):,} target records in {time.time() - t0:.1f}s.", flush=True)
    return records


def make_training_data(s1, truth, negative_ids, target_records):
    print("[6/6] Extracting 20 similarity features for training pairs...", flush=True)
    X = []
    y = []
    groups = []

    tracker = ProgressTracker(len(s1), "Feature Extraction", update_interval_sec=5)

    for row in s1.itertuples(index=False):
        s1_id = row.entity_id
        a = {
            "name_norm": normalize_name(row.business_name),
            "address_norm": normalize_address(row.business_address),
            "country_norm": normalize_text(row.country)
        }

        # Positives
        for tid in truth.get(s1_id, set()):
            if tid in target_records:
                b_row = target_records[tid]
                b = {
                    "name_norm": normalize_name(b_row.business_name),
                    "address_norm": normalize_address(b_row.business_address),
                    "country_norm": normalize_text(b_row.country)
                }
                X.append(make_features(a, b))
                y.append(1)
                groups.append(s1_id)

        # Negatives
        for tid in negative_ids[s1_id]:
            if tid in target_records:
                b_row = target_records[tid]
                b = {
                    "name_norm": normalize_name(b_row.business_name),
                    "address_norm": normalize_address(b_row.business_address),
                    "country_norm": normalize_text(b_row.country)
                }
                X.append(make_features(a, b))
                y.append(0)
                groups.append(s1_id)

        tracker.update(1)

    X_df = pd.DataFrame(X)
    y_arr = np.array(y)
    return X_df, y_arr, groups


def train_model(X, y, groups):
    print("\n==========================================", flush=True)
    print(f"TRAINING MODEL: {len(X):,} total samples ({y.sum():,} positive, {len(y) - y.sum():,} negative)", flush=True)
    print("==========================================", flush=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print("Fitting Random Forest Classifier with 300 trees...", flush=True)
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"Model fitted in {time.time() - t0:.1f} seconds.", flush=True)

    probs = model.predict_proba(X_val)[:, 1]

    print("\nValidation Results & F0.5 Threshold Tuning:")
    print("-" * 55)
    best_f05 = -1.0
    best_thresh = 0.50

    for thresh in np.arange(0.30, 0.95, 0.05):
        thresh = round(thresh, 2)
        preds = (probs >= thresh).astype(int)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f05 = fbeta_score(y_val, preds, beta=0.5, zero_division=0)

        marker = ""
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            marker = " <-- BEST"

        print(f"Threshold: {thresh:.2f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F0.5: {f05:.4f}{marker}", flush=True)

    os.makedirs("output", exist_ok=True)
    model_data = {
        "model": model,
        "best_threshold": best_thresh,
        "feature_names": list(X.columns)
    }
    joblib.dump(model_data, "output/model.joblib")
    print(f"\nSaved model and optimal threshold ({best_thresh}) to output/model.joblib", flush=True)
    return model, best_thresh


def main():
    t_start = time.time()
    s1, gt = load_data()
    truth = build_ground_truth(gt)
    s1_sampled = sample_source1(s1, truth)
    block_to_s1 = build_block_index(s1_sampled)
    negative_ids, positive_ids = find_candidate_ids(s1_sampled, truth, block_to_s1)
    target_records = load_required_targets(positive_ids, negative_ids)
    X, y, groups = make_training_data(s1_sampled, truth, negative_ids, target_records)
    train_model(X, y, groups)
    print(f"\nPipeline completed in {time.time() - t_start:.1f} seconds!", flush=True)


if __name__ == "__main__":
    main()