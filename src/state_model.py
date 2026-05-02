# src/state_model.py
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def split_by_subject(
    meta_all: List[Dict[str, Any]],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    subj_list = sorted({m["subject"] for m in meta_all})
    rng = np.random.RandomState(random_state)
    rng.shuffle(subj_list)

    n = len(subj_list)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    train_subj = subj_list[:n_train]
    val_subj = subj_list[n_train:n_train + n_val]
    test_subj = subj_list[n_train + n_val:]

    train_idx, val_idx, test_idx = [], [], []
    for i, m in enumerate(meta_all):
        s = m["subject"]
        if s in train_subj:
            train_idx.append(i)
        elif s in val_subj:
            val_idx.append(i)
        else:
            test_idx.append(i)

    return {
        "train_idx": np.array(train_idx, dtype=int),
        "val_idx": np.array(val_idx, dtype=int),
        "test_idx": np.array(test_idx, dtype=int),
        "train_subjects": train_subj,
        "val_subjects": val_subj,
        "test_subjects": test_subj,
    }


def train_state_classifier(X, y, random_state: int = 42):
    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state
    )
    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_te)
    print("=== State Classification Report ===")
    print(classification_report(y_te, y_pred))
    return clf


def train_state_classifier_subject_split(X, y, meta, random_state: int = 42):
    split = split_by_subject(
        meta, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        random_state=random_state
    )
    idx_train = np.concatenate([split["train_idx"], split["val_idx"]], axis=0)
    idx_test = split["test_idx"]

    x_tr, y_tr = X[idx_train], y[idx_train]
    x_te, y_te = X[idx_test], y[idx_test]

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state
    )
    clf.fit(x_tr, y_tr)

    y_pred = clf.predict(x_te)
    print("=== State Classification Report (subject-level split) ===")
    print(classification_report(y_te, y_pred))

    return clf, y_te, y_pred, idx_test, split