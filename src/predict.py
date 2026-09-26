import os
import sys
import time
import joblib
import multiprocessing as mp
import pandas as pd

try:
    from .blocking import get_blocks
    from .features import make_features
    from .utils import normalize_name, normalize_address, normalize_text
except ImportError:
    from blocking import get_blocks
    from features import make_features
    from utils import normalize_name, normalize_address, normalize_text

MODEL_PATH = "output/model.joblib"

S1_FILE = "dataset/test/test_source1.tsv"
S2_FILE = "dataset/test/test_source2.tsv"
S3_FILE = "dataset/test/test_source3.tsv"

MATCHING_FILE = "output/matching_results.tsv"
CANDIDATE_FILE = "output/candidate_pairs.tsv"

CHUNK_SIZE = 100000
SCORE_BATCH_SIZE = 100000
THRESHOLD = 0.80
MAX_BLOCK_SIZE = 15

# Global variable for worker processes
_GLOBAL_BLOCK_TO_S1 = None


def p(msg):
    print(msg, flush=True)


def init_worker(block_to_s1):
    global _GLOBAL_BLOCK_TO_S1
    _GLOBAL_BLOCK_TO_S1 = block_to_s1


def process_lines_chunk(lines_chunk):
    global _GLOBAL_BLOCK_TO_S1
    block_to_s1 = _GLOBAL_BLOCK_TO_S1

    hits = []
    target_fields = {}

    for line in lines_chunk:
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
            target_fields[tid] = (b_name, b_addr, b_ctry)
            for s1_id in matched_s1:
                hits.append((s1_id, tid))

    return hits, target_fields


def make_features_wrapper(pair):
    a, b = pair
    return make_features(a, b)


def main():
    t_start = time.time()
    num_cpus = os.cpu_count() or 4
    p(f"Starting multi-core prediction pipeline with {num_cpus} processes...")

    if not os.path.exists(MODEL_PATH):
        p(f"Model file not found at {MODEL_PATH}. Training model first...")
        from train import main as train_main
        train_main()

    p("Loading ML model...")
    model = joblib.load(MODEL_PATH)

    p("Loading Source 1 test dataset...")
    s1_df = pd.read_csv(
        S1_FILE,
        sep="\t",
        encoding="latin1",
        dtype=str
    ).fillna("")

    s1_ids = s1_df["entity_id"].tolist()
    p(f"Total Source 1 entities: {len(s1_ids)}")

    p("Building block index...")
    s1_raw = {}
    block_to_s1 = {}

    for row in s1_df.itertuples(index=False):
        eid = row.entity_id
        s1_raw[eid] = (row.business_name, row.business_address, row.country)

        keys = get_blocks(
            row.business_name,
            row.business_address,
            row.country
        )
        for k in keys:
            if k not in block_to_s1:
                block_to_s1[k] = []
            block_to_s1[k].append(eid)

    p(f"Total unique block keys generated: {len(block_to_s1)}")

    oversized = [k for k, ids in block_to_s1.items() if len(ids) > MAX_BLOCK_SIZE]
    if oversized:
        p(f"Pruning {len(oversized)} block key(s) exceeding max block size of {MAX_BLOCK_SIZE}.")
        for k in oversized:
            del block_to_s1[k]

    p("\nScanning target files in parallel across all CPU cores...")
    candidates_map = {eid: [] for eid in s1_ids}
    target_raw = {}
    total_candidate_pairs = 0

    with mp.Pool(processes=num_cpus, initializer=init_worker, initargs=(block_to_s1,)) as pool:
        for file_path in [S2_FILE, S3_FILE]:
            if not os.path.exists(file_path):
                p(f"Warning: file {file_path} not found, skipping...")
                continue

            p(f"Reading: {file_path}")
            processed_lines = 0

            with open(file_path, "r", encoding="latin1", errors="ignore") as f:
                header = f.readline()  # Skip header

                chunk_batch = []
                current_chunk = []
                batch_line_count = 0

                for line in f:
                    current_chunk.append(line)
                    if len(current_chunk) >= CHUNK_SIZE:
                        chunk_batch.append(current_chunk)
                        batch_line_count += len(current_chunk)
                        current_chunk = []

                    # Process 8 chunks (800k lines) in parallel per pool dispatch
                    if len(chunk_batch) >= num_cpus:
                        results = pool.map(process_lines_chunk, chunk_batch)
                        for hits, target_fields in results:
                            target_raw.update(target_fields)
                            for s1_id, tid in hits:
                                candidates_map[s1_id].append(tid)
                                total_candidate_pairs += 1
                        processed_lines += batch_line_count
                        p(f"  Processed {processed_lines} lines | Candidate pairs: {total_candidate_pairs}")
                        chunk_batch = []
                        batch_line_count = 0

                if current_chunk:
                    chunk_batch.append(current_chunk)
                    batch_line_count += len(current_chunk)

                if chunk_batch:
                    results = pool.map(process_lines_chunk, chunk_batch)
                    for hits, target_fields in results:
                        target_raw.update(target_fields)
                        for s1_id, tid in hits:
                            candidates_map[s1_id].append(tid)
                            total_candidate_pairs += 1
                    processed_lines += batch_line_count
                    p(f"  Processed {processed_lines} lines | Candidate pairs: {total_candidate_pairs}")

            p(f"  Finished {file_path} | Total lines: {processed_lines} | Candidate pairs: {total_candidate_pairs}")

    p(f"\nCandidate generation complete.")
    p(f"Total candidate pairs: {total_candidate_pairs}")

    p("\nScoring candidates in parallel batches...")
    matched_map = {eid: [] for eid in s1_ids}

    s1_norm_cache = {}
    target_norm_cache = {}

    batch_pairs = []
    batch_pair_ids = []
    total_scored = 0

    with mp.Pool(processes=num_cpus) as scoring_pool:
        for s1_id in s1_ids:
            cands = candidates_map[s1_id]
            if not cands:
                continue

            unique_cands = list(dict.fromkeys(cands))
            candidates_map[s1_id] = unique_cands

            if s1_id not in s1_norm_cache:
                b_name, b_addr, b_ctry = s1_raw[s1_id]
                s1_norm_cache[s1_id] = {
                    "name_norm": normalize_name(b_name),
                    "address_norm": normalize_address(b_addr),
                    "country_norm": normalize_text(b_ctry)
                }
            a = s1_norm_cache[s1_id]

            for tid in unique_cands:
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
                    feats = scoring_pool.map(make_features_wrapper, batch_pairs, chunksize=5000)
                    X = pd.DataFrame(feats)
                    probs = model.predict_proba(X)[:, 1]
                    for (s_id, t_id), prob in zip(batch_pair_ids, probs):
                        if prob >= THRESHOLD:
                            matched_map[s_id].append(t_id)

                    total_scored += len(batch_pairs)
                    p(f"  Scored {total_scored} pairs...")
                    batch_pairs = []
                    batch_pair_ids = []

        if batch_pairs:
            feats = scoring_pool.map(make_features_wrapper, batch_pairs, chunksize=5000)
            X = pd.DataFrame(feats)
            probs = model.predict_proba(X)[:, 1]
            for (s_id, t_id), prob in zip(batch_pair_ids, probs):
                if prob >= THRESHOLD:
                    matched_map[s_id].append(t_id)

            total_scored += len(batch_pairs)
            p(f"  Scored {total_scored} pairs total.")

    p("Scoring complete.")

    p("\nWriting output TSV files...")
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
    p("==================================")
    p(f"PREDICTION PIPELINE COMPLETED IN {t_end - t_start:.2f} SECONDS")
    p("==================================")


if __name__ == "__main__":
    main()