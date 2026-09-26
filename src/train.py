import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, fbeta_score

from utils import normalize_name, normalize_address, normalize_text
from features import make_features
from blocking import get_blocks


S1_SAMPLE_SIZE = 20000
CHUNK_SIZE = 100000
NEGATIVES_PER_S1 = 10

RAW_COLUMNS = ["business_name", "business_address", "country"]


def fill_missing(df):
    # pd.read_csv(..., dtype=str) still yields NaN (a float) for empty
    # cells. Filling these to "" right after reading keeps every
    # downstream consumer (blocking, feature building, SQLite storage)
    # from ever seeing a NaN and stringifying it into the literal "nan".
    df[RAW_COLUMNS] = df[RAW_COLUMNS].fillna("")
    return df


def load_data():

    s1 = pd.read_csv(
        "dataset/train/train_source1.tsv",
        sep="\t",
        encoding="latin1",
        dtype=str
    )

    s1 = fill_missing(s1)

    gt = pd.read_csv(
        "dataset/train/train_ground_truth.tsv",
        sep="\t",
        encoding="latin1",
        dtype=str
    )

    return s1, gt


def build_ground_truth(gt):

    truth = {}

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]

        value = row["matched_entity_ids"]

        if pd.isna(value):
            value = ""

        ids = set()

        if str(value).strip():

            ids = set(
                x.strip()
                for x in str(value).split(",")
                if x.strip()
            )

        truth[s1_id] = ids

    return truth


def sample_source1(s1, truth):

    matched = []
    unmatched = []

    for _, row in s1.iterrows():

        entity_id = row["entity_id"]

        if truth.get(entity_id, set()):
            matched.append(entity_id)
        else:
            unmatched.append(entity_id)

    rng = np.random.default_rng(42)

    n_matched = min(
        S1_SAMPLE_SIZE // 2,
        len(matched)
    )

    n_unmatched = min(
        S1_SAMPLE_SIZE - n_matched,
        len(unmatched)
    )

    selected = []

    selected.extend(
        rng.choice(
            matched,
            size=n_matched,
            replace=False
        )
    )

    selected.extend(
        rng.choice(
            unmatched,
            size=n_unmatched,
            replace=False
        )
    )

    selected = set(selected)

    s1 = s1[
        s1["entity_id"].isin(selected)
    ].copy()

    print(
        "Training Source1:",
        len(s1)
    )

    print(
        "Matched Source1:",
        n_matched
    )

    print(
        "Unmatched Source1:",
        n_unmatched
    )

    return s1


def build_block_index(s1):

    block_to_s1 = {}

    for _, row in s1.iterrows():

        entity_id = row["entity_id"]

        keys = get_blocks(
            row["business_name"],
            row["business_address"],
            row["country"]
        )

        for key in keys:

            if key not in block_to_s1:
                block_to_s1[key] = []

            block_to_s1[key].append(
                entity_id
            )

    print(
        "Source1 blocks:",
        len(block_to_s1)
    )

    return block_to_s1


def get_positive_ids(s1, truth):

    positive_ids = set()

    for entity_id in s1["entity_id"]:

        positive_ids.update(
            truth.get(entity_id, set())
        )

    print(
        "Positive target IDs:",
        len(positive_ids)
    )

    return positive_ids


def find_candidate_ids(
    s1,
    truth,
    block_to_s1
):

    negative_ids = {
        entity_id: []
        for entity_id in s1["entity_id"]
    }

    positive_ids = get_positive_ids(
        s1,
        truth
    )

    files = [
        "dataset/train/train_source2.tsv",
        "dataset/train/train_source3.tsv"
    ]

    print("Searching target candidates...")

    for file in files:

        print("Reading:", file)

        for chunk in pd.read_csv(
            file,
            sep="\t",
            encoding="latin1",
            dtype=str,
            chunksize=CHUNK_SIZE
        ):

            chunk = fill_missing(chunk)

            for row in chunk.itertuples(
                index=False
            ):

                target_id = row.entity_id

                keys = get_blocks(
                    row.business_name,
                    row.business_address,
                    row.country
                )

                matched_source1 = set()

                for key in keys:

                    if key in block_to_s1:

                        matched_source1.update(
                            block_to_s1[key]
                        )

                for s1_id in matched_source1:

                    if target_id in truth.get(
                        s1_id,
                        set()
                    ):
                        continue

                    current = negative_ids[s1_id]

                    if len(current) < NEGATIVES_PER_S1:

                        current.append(
                            target_id
                        )

            print(
                "Processed chunk:",
                len(chunk)
            )

    print("Candidate search complete.")

    return negative_ids, positive_ids


def load_required_targets(
    positive_ids,
    negative_ids
):

    required_ids = set(
        positive_ids
    )

    for ids in negative_ids.values():
        required_ids.update(ids)

    print(
        "Required target records:",
        len(required_ids)
    )

    records = {}

    files = [
        "dataset/train/train_source2.tsv",
        "dataset/train/train_source3.tsv"
    ]

    for file in files:

        if len(records) == len(required_ids):
            break

        print(
            "Loading required records from:",
            file
        )

        for chunk in pd.read_csv(
            file,
            sep="\t",
            encoding="latin1",
            dtype=str,
            chunksize=CHUNK_SIZE
        ):

            chunk = fill_missing(chunk)

            selected = chunk[
                chunk["entity_id"].isin(
                    required_ids
                )
            ]

            for row in selected.itertuples(
                index=False
            ):

                records[row.entity_id] = row

            if len(records) == len(required_ids):
                break

    print(
        "Target records loaded:",
        len(records)
    )

    return records


def make_training_data(
    s1,
    truth,
    negative_ids,
    target_records
):

    X = []
    y = []
    groups = []

    print("Creating features...")

    for count, row in enumerate(
        s1.itertuples(index=False)
    ):

        s1_id = row.entity_id

        a = {
            "name_norm": normalize_name(
                row.business_name
            ),
            "address_norm": normalize_address(
                row.business_address
            ),
            "country_norm": normalize_text(
                row.country
            )
        }

        positive_ids = truth.get(
            s1_id,
            set()
        )

        for target_id in positive_ids:

            if target_id not in target_records:
                continue

            b_row = target_records[target_id]

            b = {
                "name_norm": normalize_name(
                    b_row.business_name
                ),
                "address_norm": normalize_address(
                    b_row.business_address
                ),
                "country_norm": normalize_text(
                    b_row.country
                )
            }

            X.append(
                make_features(a, b)
            )

            y.append(1)
            groups.append(s1_id)

        for target_id in negative_ids[s1_id]:

            if target_id not in target_records:
                continue

            b_row = target_records[target_id]

            b = {
                "name_norm": normalize_name(
                    b_row.business_name
                ),
                "address_norm": normalize_address(
                    b_row.business_address
                ),
                "country_norm": normalize_text(
                    b_row.country
                )
            }

            X.append(
                make_features(a, b)
            )

            y.append(0)
            groups.append(s1_id)

        if (count + 1) % 2000 == 0:

            print(
                "Processed:",
                count + 1,
                "/",
                len(s1),
                "Training rows:",
                len(X)
            )

    X = pd.DataFrame(X)
    y = np.array(y)

    return X, y, groups


def train_model(X, y, groups):

    print(
        "Training rows:",
        len(X)
    )

    print(
        "Positive matches:",
        int(y.sum())
    )

    print(
        "Negative matches:",
        int(len(y) - y.sum())
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    print(
        "Training model..."
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    print()
    print("Validation results")
    print("------------------")

    for threshold in [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95
    ]:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        f05 = fbeta_score(
            y_val,
            predictions,
            beta=0.5,
            zero_division=0
        )

        print(
            "Threshold:",
            threshold,
            "Precision:",
            round(precision, 4),
            "Recall:",
            round(recall, 4),
            "F0.5:",
            round(f05, 4)
        )

    joblib.dump(
        model,
        "output/model.joblib"
    )

    print()
    print("Model saved to output/model.joblib")

    return model


def main():

    s1, gt = load_data()

    truth = build_ground_truth(gt)

    s1 = sample_source1(
        s1,
        truth
    )

    block_to_s1 = build_block_index(
        s1
    )

    negative_ids, positive_ids = find_candidate_ids(
        s1,
        truth,
        block_to_s1
    )

    target_records = load_required_targets(
        positive_ids,
        negative_ids
    )

    X, y, groups = make_training_data(
        s1,
        truth,
        negative_ids,
        target_records
    )

    train_model(
        X,
        y,
        groups
    )


if __name__ == "__main__":                  
    main()