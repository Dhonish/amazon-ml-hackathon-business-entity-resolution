# Business Entity Resolution Challenge — Scalable ML Pipeline

This repository contains an end-to-end Machine Learning solution for the **Amazon ML Challenge 2026: Business Entity Resolution**. 
The pipeline resolves business entities across noisy, multi-source dataset fragments (`Source 1`, `Source 2`, `Source 3`) with strict scalability and high precision constraints.

---

## Technical Overview & Methodology

### 1. Blocking / Candidate Generation
Comparing every record with every other record across ~11 million total entities is computationally infeasible ($O(N \times M)$). To make entity resolution scale:
- **Multi-Key Blocking**: We generate country-scoped indexing keys based on exact normalized business names, name prefix tokens, and address tokens (`blocking.py`).
- **Selective Hit Thresholding**: A candidate target record is paired with a Source 1 entity only if they share at least **2 independent block keys** or have an **exact normalized name match**.
- **Oversized Block Pruning**: Generic block keys returning >50 matches are pruned to prevent search space explosion.
- **Candidate Efficiency**: This reduces the candidate space per Source 1 entity to a minimal candidate set while maintaining high recall ceiling, directly fulfilling the candidate generation evaluation criteria for `candidate_pairs.tsv`.

### 2. Feature Engineering
For candidate pair $(A, B)$, 11 distinct similarity features are generated (`features.py`):
- **Name Similarities**: RapidFuzz Ratio, Token Set Ratio, Weighted Ratio (WRatio), and Jaccard word similarity.
- **Address Similarities**: RapidFuzz Ratio, Token Set Ratio, WRatio, and Jaccard similarity.
- **Exact & Categorical Match Flags**: Country matching indicator, exact name match flag, and exact address match flag.

### 3. Model Architecture & Optimization
- **Classifier**: `RandomForestClassifier` with balanced class weights, tree depth constraints (`max_depth=12`), and `n_estimators=150`.
- **Metric Alignment**: The decision threshold is tuned for the **$F_{0.5}$ score**, which weights precision twice as heavily as recall to penalize false entity merges.

---

## Directory Structure

```
.
├── dataset/
│   ├── train/               # Ground truth and training sources
│   └── test/                # Test sources (test_source1, test_source2, test_source3)
├── output/
│   ├── matching_results.tsv  # Final entity matches (scored on leaderboard)
│   ├── candidate_pairs.tsv   # Blocking candidate set
│   └── model.joblib          # Trained Random Forest model checkpoint
├── src/
│   ├── blocking.py          # Multi-key candidate generation & blocking logic
│   ├── features.py          # RapidFuzz and Jaccard feature extraction
│   ├── train.py             # Model training script
│   ├── predict.py           # Scalable streaming candidate generation & prediction
│   └── utils.py             # Text, name, and address normalization
├── utils/
│   └── validate_submission.py  # Local submission validation tool
├── README.md
└── requirements.txt
```

---

## Reproducing Results

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
To retrain the ML model on the ground truth training set:
```bash
python src/train.py
```
This generates `output/model.joblib`.

### 3. Run Candidate Generation & Prediction
To generate `output/matching_results.tsv` and `output/candidate_pairs.tsv` for all test entities:
```bash
python src/predict.py
```

### 4. Validate Submission
Verify format compliance using the provided validator:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
