import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
import joblib
import warnings
import multiprocessing as mp
import pandas as pd

warnings.filterwarnings("ignore")

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

SCORE_BATCH_SIZE = 50000
MAX_BLOCK_SIZE = 35
MAX_CANDIDATES_PER_S1 = 15
DEFAULT_THRESHOLD = 0.80


def p(msg):
    print(msg, flush=True)


def score_worker_chunk(chunk_args):
    s1_chunk_ids, s1_norm_chunk, candidates_map_chunk, target_norm_chunk, model_path, threshold = chunk_args

    model_obj = joblib.load(model_path)
    if isinstance(model_obj, dict):
        model = model_obj["model"]
        feature_names = model_obj.get("feature_names", None)
    else:
        model = model_obj
        feature_names = None

    local_matched = {}
    batch_feats = []
    batch_pair_ids = []

    def run_batch():
        nonlocal batch_feats, batch_pair_ids
        if not batch_feats:
            return
        X_df = pd.DataFrame(batch_feats)
        if feature_names:
            X_df = X_df[feature_names]
        probs = model.predict_proba(X_df)[:, 1]

        for (s_id, t_id), prob in zip(batch_pair_ids, probs):
            if prob >= threshold:
                local_matched.setdefault(s_id, []).append(t_id)

        batch_feats = []
        batch_pair_ids = []

    for s1_id in s1_chunk_ids:
        cands = candidates_map_chunk.get(s1_id, [])
        if not cands:
            continue

        a = s1_norm_chunk[s1_id]
        for tid in cands:
            if tid not in target_norm_chunk:
                continue
            b = target_norm_chunk[tid]
            feats = make_features(a, b)

            batch_feats.append(feats)
            batch_pair_ids.append((s1_id, tid))

            if len(batch_feats) >= SCORE_BATCH_SIZE:
                run_batch()

    run_batch()
    return local_matched


def main():
    t_start = time.time()
    num_workers = min(os.cpu_count() or 4, 8)
    p("==================================================")
    p("STARTING PARALLEL HIGH-PRECISION PREDICTION PIPELINE")
    p(f"CPU Workers: {num_workers} | Max Block Size: {MAX_BLOCK_SIZE} | Max Candidates/S1: {MAX_CANDIDATES_PER_S1}")
    p("==================================================")

    if not os.path.exists(MODEL_PATH):
        p(f"Model file not found at {MODEL_PATH}. Running model training first...")
        from train import main as train_main
        train_main()

    p("\n[1/5] Loading trained 20-feature ML model info...")
    model_obj = joblib.load(MODEL_PATH)
    if isinstance(model_obj, dict):
        model = model_obj["model"]
        threshold = model_obj.get("best_threshold", DEFAULT_THRESHOLD)
    else:
        model = model_obj
        threshold = DEFAULT_THRESHOLD

    p(f"Model loaded successfully (n_features_in_={model.n_features_in_}). Decision Threshold: {threshold:.2f}")

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
    s1_norm = {}
    block_to_s1 = {}
    tracker = ProgressTracker(len(s1_ids), "S1 Pre-processing", update_interval_sec=10)

    for row in s1_df.itertuples(index=False):
        eid = row.entity_id
        b_name = normalize_name(row.business_name)
        b_addr = normalize_address(row.business_address)
        b_ctry = normalize_text(row.country)

        s1_norm[eid] = {
            "name_norm": b_name,
            "address_norm": b_addr,
            "country_norm": b_ctry
        }

        keys = get_blocks(row.business_name, row.business_address, row.country)
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
    candidates_scores = {eid: {} for eid in s1_ids}
    target_norm = {}
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
                    target_norm[tid] = {
                        "name_norm": normalize_name(b_name),
                        "address_norm": normalize_address(b_addr),
                        "country_norm": normalize_text(b_ctry)
                    }
                    for s1_id in matched_s1:
                        scores = candidates_scores[s1_id]
                        scores[tid] = scores.get(tid, 0) + 1
                        raw_hits += 1

                if processed_lines % 1000000 == 0:
                    elapsed = time.time() - t_scan_start
                    speed = processed_lines / elapsed if elapsed > 0 else 0
                    p(f"  [Target Scan] Processed {processed_lines:,} lines | Speed: {speed:,.0f} lines/s | Raw Hits: {raw_hits:,}")

        p(f"  Finished {file_path} | Lines: {processed_lines:,}")

    p("\nPriority-ranking candidates per S1 entity by blocking hit strength...")
    candidates_map = {}
    total_candidate_pairs = 0

    for s1_id in s1_ids:
        scores = candidates_scores[s1_id]
        if scores:
            sorted_cands = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)[:MAX_CANDIDATES_PER_S1]
            candidates_map[s1_id] = sorted_cands
            total_candidate_pairs += len(sorted_cands)
        else:
            candidates_map[s1_id] = []

    del candidates_scores

    p(f"Candidate Generation Complete. Total Priority-Ranked Pairs to Score: {total_candidate_pairs:,}")

    p(f"\n[5/5] Parallel Feature Extraction & Scoring across {num_workers} worker processes...")
    chunk_size = (len(s1_ids) + num_workers - 1) // num_workers
    worker_tasks = []

    for i in range(num_workers):
        chunk_s1_ids = s1_ids[i * chunk_size: (i + 1) * chunk_size]
        if not chunk_s1_ids:
            continue

        s1_norm_chunk = {eid: s1_norm[eid] for eid in chunk_s1_ids}
        candidates_map_chunk = {eid: candidates_map[eid] for eid in chunk_s1_ids}

        target_ids_needed = set()
        for cands in candidates_map_chunk.values():
            target_ids_needed.update(cands)

        target_norm_chunk = {tid: target_norm[tid] for tid in target_ids_needed if tid in target_norm}

        worker_tasks.append((
            chunk_s1_ids,
            s1_norm_chunk,
            candidates_map_chunk,
            target_norm_chunk,
            MODEL_PATH,
            threshold
        ))

    del s1_norm
    del target_norm

    matched_map = {}

    with mp.Pool(processes=len(worker_tasks)) as pool:
        results = pool.map(score_worker_chunk, worker_tasks)
        for res in results:
            matched_map.update(res)

    p("\nWriting final submission TSV files...")
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