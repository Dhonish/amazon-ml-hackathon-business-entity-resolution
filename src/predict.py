import os
import sys
import time
import joblib
import multiprocessing as mp
import pandas as pd

try:
    from .blocking import get_blocks
    from .features import make_features
    from .utils import normalize_name, normalize_address, normalize_text, ProgressTracker
except ImportError:
    from blocking import get_blocks
    from features import make_features
    from utils import normalize_name, normalize_address, normalize_text, ProgressTracker

MODEL_PATH = "output/model.joblib"

S1_FILE = "dataset/test/test_source1.tsv"
S2_FILE = "dataset/test/test_source2.tsv"
S3_FILE = "dataset/test/test_source3.tsv"

MATCHING_FILE = "output/matching_results.tsv"
CANDIDATE_FILE = "output/candidate_pairs.tsv"

SCORE_BATCH_SIZE = 200000
MAX_BLOCK_SIZE = 40
MAX_CANDIDATES_PER_S1 = 25


def p(msg):
    print(msg, flush=True)


def make_features_wrapper(pair):
    a, b = pair
    return make_features(a, b)


def main():
    t_start = time.time()
    num_cpus = os.cpu_count() or 4
    p("==================================================")
    p("STARTING OPTIMIZED HIGH-RECALL PREDICTION PIPELINE")
    p(f"CPU Cores: {num_cpus} | Max Block Size: {MAX_BLOCK_SIZE} | Max Candidates/S1: {MAX_CANDIDATES_PER_S1}")
    p("==================================================")

    if not os.path.exists(MODEL_PATH):
        p(f"Model file not found at {MODEL_PATH}. Running model training first...")
        from train import main as train_main
        train_main()

    p("\n[1/5] Loading trained ML model & decision threshold...")
    model_obj = joblib.load(MODEL_PATH)

    if isinstance(model_obj, dict):
        model = model_obj["model"]
        threshold = model_obj.get("best_threshold", 0.85)
    else:
        model = model_obj
        threshold = 0.85

    p(f"Model loaded successfully. Optimal F0.5 Threshold: {threshold:.2f}")

    p("\n[2/5] Loading Source 1 test dataset...")
    s1_df = pd.read_csv(
        S1_FILE,
        sep="\t",
        encoding="latin1",
        dtype=str
    ).fillna("")

    s1_ids = s1_df["entity_id"].tolist()
    p(f"Total Source 1 test entities: {len(s1_ids):,}")

    p("\n[3/5] Pre-normalizing S1 entities and building multi-key block index...")
    s1_raw = {}
    block_to_s1 = {}
    tracker = ProgressTracker(len(s1_ids), "S1 Pre-processing", update_interval_sec=10)

    for row in s1_df.itertuples(index=False):
        eid = row.entity_id
        s1_raw[eid] = (row.business_name, row.business_address, row.country)

        keys = get_blocks(
            row.business_name,
            row.business_address,
            row.country
        )
        for k in keys:
            block_to_s1.setdefault(k, []).append(eid)
        tracker.update(1)

    p(f"Total unique block keys generated: {len(block_to_s1):,}")

    oversized = [k for k, ids in block_to_s1.items() if len(ids) > MAX_BLOCK_SIZE]
    if oversized:
        p(f"Pruning {len(oversized):,} generic block key(s) exceeding max block size of {MAX_BLOCK_SIZE}.")
        for k in oversized:
            del block_to_s1[k]

    p("\n[4/5] Streaming target dataset scanning (test_source2 & test_source3)...")
    candidates_map = {eid: [] for eid in s1_ids}
    target_raw = {}
    raw_hits = 0
    t_scan_start = time.time()

    for file_path in [S2_FILE, S3_FILE]:
        if not os.path.exists(file_path):
            p(f"Warning: File {file_path} not found, skipping...")
            continue

        p(f"Scanning target file: {file_path}")
        processed_lines = 0

        with open(file_path, "r", encoding="latin1", errors="ignore") as f:
            f.readline()  # Skip header

            for line in f:
                processed_lines += 1
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue

                tid, b_name, b_addr, b_ctry = parts[0], parts[1], parts[2], parts[3]
                keys = get_blocks(b_name, b_addr, b_ctry)

                matched_s1 = set()
                for k in keys:
                    if k in block_to_s1:
                        matched_s1.update(block_to_s1[k])

                if matched_s1:
                    target_raw[tid] = (b_name, b_addr, b_ctry)
                    for s1_id in matched_s1:
                        candidates_map[s1_id].append(tid)
                        raw_hits += 1

                if processed_lines % 1000000 == 0:
                    elapsed = time.time() - t_scan_start
                    speed = processed_lines / elapsed if elapsed > 0 else 0
                    p(f"  [Target Scan] Processed {processed_lines:,} lines | Speed: {speed:,.0f} lines/s | Raw Hits: {raw_hits:,}")

        p(f"  Finished {file_path} | Lines: {processed_lines:,}")

    p("\nOptimizing candidate pairs per S1 entity...")
    total_candidate_pairs = 0
    for s1_id in s1_ids:
        cands = candidates_map[s1_id]
        if cands:
            unique_cands = list(dict.fromkeys(cands))[:MAX_CANDIDATES_PER_S1]
            candidates_map[s1_id] = unique_cands
            total_candidate_pairs += len(unique_cands)

    p(f"Candidate Generation Complete. Total Candidate Pairs to Score: {total_candidate_pairs:,}")

    p("\n[5/5] Parallel Feature Extraction & Model Scoring across CPU cores...")
    matched_map = {eid: [] for eid in s1_ids}

    s1_norm_cache = {}
    target_norm_cache = {}

    batch_pairs = []
    batch_pair_ids = []

    score_tracker = ProgressTracker(total_candidate_pairs, "Pair Scoring", update_interval_sec=10)

    with mp.Pool(processes=num_cpus) as scoring_pool:
        for s1_id in s1_ids:
            cands = candidates_map[s1_id]
            if not cands:
                continue

            if s1_id not in s1_norm_cache:
                b_name, b_addr, b_ctry = s1_raw[s1_id]
                s1_norm_cache[s1_id] = {
                    "name_norm": normalize_name(b_name),
                    "address_norm": normalize_address(b_addr),
                    "country_norm": normalize_text(b_ctry)
                }
            a = s1_norm_cache[s1_id]

            for tid in cands:
                if tid not in target_norm_cache:
                    tb_name, tb_addr, tb_ctry = target_raw[tid]
                    target_norm_cache[tid] = {
                        "name_norm": normalize_name(tb_name),
                        "address_norm": normalize_address(tb_addr),
                        "country_norm": normalize_text(tb_ctry)
                    }
                b = target_norm_cache[tid]

                batch_pairs.append((a, b))
                batch_pair_ids.append((s1_id, tid))

                if len(batch_pairs) >= SCORE_BATCH_SIZE:
                    feats = scoring_pool.map(make_features_wrapper, batch_pairs, chunksize=10000)
                    X = pd.DataFrame(feats)
                    probs = model.predict_proba(X)[:, 1]
                    for (s_id, t_id), prob in zip(batch_pair_ids, probs):
                        if prob >= threshold:
                            matched_map[s_id].append(t_id)

                    score_tracker.update(len(batch_pairs))
                    batch_pairs = []
                    batch_pair_ids = []

        if batch_pairs:
            feats = scoring_pool.map(make_features_wrapper, batch_pairs, chunksize=10000)
            X = pd.DataFrame(feats)
            probs = model.predict_proba(X)[:, 1]
            for (s_id, t_id), prob in zip(batch_pair_ids, probs):
                if prob >= threshold:
                    matched_map[s_id].append(t_id)

            score_tracker.update(len(batch_pairs))

    p("\nWriting final output TSV files...")
    os.makedirs("output", exist_ok=True)

    with open(MATCHING_FILE, "w", encoding="utf-8") as fm, open(CANDIDATE_FILE, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in s1_ids:
            cands = candidates_map.get(s1_id, [])
            matches = matched_map.get(s1_id, [])

            c_str = ",".join(cands)
            m_str = ",".join(matches)

            fc.write(f"{s1_id}\t{c_str}\n")
            fm.write(f"{s1_id}\t{m_str}\n")

    t_end = time.time()
    p(f"Saved matching results to: {MATCHING_FILE}")
    p(f"Saved candidate pairs to: {CANDIDATE_FILE}")
    p("==================================================")
    p(f"PREDICTION PIPELINE COMPLETED IN {(t_end - t_start) / 60:.2f} MINUTES")
    p("==================================================")


if __name__ == "__main__":
    main()