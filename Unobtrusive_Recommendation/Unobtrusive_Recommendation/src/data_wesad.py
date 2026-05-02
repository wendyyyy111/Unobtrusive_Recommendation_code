# src/data_wesad.py
import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import DATA_DIR, WIN_SEC, STEP_SEC, FS_CHEST_DEFAULT


def load_subject_pkl(subject_id: int, data_dir: str = DATA_DIR):
    pkl_path = os.path.join(data_dir, f"S{subject_id}", f"S{subject_id}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Cannot find: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def load_subject_quest(subject_id: int, data_dir: str = DATA_DIR):
    csv_path = os.path.join(data_dir, f"S{subject_id}", f"S{subject_id}_quest.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find: {csv_path}")
    return pd.read_csv(csv_path, header=None)


def inspect_subject_structure(data: Dict[str, Any], subject_id: int):
    print(f"\n===== Inspect S{subject_id} structure =====")
    sig = data["signal"]
    chest = sig["chest"]
    wrist = sig["wrist"]

    print("Chest keys:", list(chest.keys()))
    for k, v in chest.items():
        print(f"  chest[{k}] shape:", np.array(v).shape)

    print("Wrist keys:", list(wrist.keys()))
    for k, v in wrist.items():
        print(f"  wrist[{k}] shape:", np.array(v).shape)

    print("Label shape:", np.array(data["label"]).shape)
    print("====================================\n")


def infer_fs_from_length(labels, total_seconds_approx: float = 7200):
    n = len(labels)
    fs_est = n / float(total_seconds_approx)
    print(f"[INFO] Approximate Fs from length: {fs_est:.2f} Hz")
    return fs_est


def _time_str_to_sec(t: str) -> int:
    t = str(t).strip()
    if not t:
        return 0
    parts = t.split(".")
    if len(parts) == 1:
        m = int(parts[0])
        s = 0
    else:
        m = int(parts[0])
        s_str = parts[1]
        if len(s_str) == 1:
            s_str = "0" + s_str
        s = int(s_str)
    return m * 60 + s


def _trim_tail_empty(lst: List[str]) -> List[str]:
    while len(lst) > 0 and (lst[-1] is None or str(lst[-1]).strip() == ""):
        lst.pop()
    return lst


def parse_condition_intervals_from_quest(df_quest: pd.DataFrame):
    row1 = str(df_quest.iloc[1, 0])
    row2 = str(df_quest.iloc[2, 0])
    row3 = str(df_quest.iloc[3, 0])

    parts1 = [p.strip() for p in row1.split(";")]
    parts2 = [p.strip() for p in row2.split(";")]
    parts3 = [p.strip() for p in row3.split(";")]

    cond_names = _trim_tail_empty(parts1[1:])
    cond_starts_raw = _trim_tail_empty(parts2[1:])
    cond_ends_raw = _trim_tail_empty(parts3[1:])

    l = min(len(cond_names), len(cond_starts_raw), len(cond_ends_raw))
    cond_names = cond_names[:l]
    cond_starts_raw = cond_starts_raw[:l]
    cond_ends_raw = cond_ends_raw[:l]

    cond_starts_sec = [_time_str_to_sec(t) for t in cond_starts_raw]
    cond_ends_sec = [_time_str_to_sec(t) for t in cond_ends_raw]

    name_to_label = {
        "base": 1,
        "tsst": 2,
        "fun": 3,
        "medi 1": 4,
        "medi1": 4,
        "medi 2": 4,
        "medi2": 4,
    }

    intervals = []
    for name, st, ed in zip(cond_names, cond_starts_sec, cond_ends_sec):
        n = name.strip().lower()
        if n not in name_to_label:
            print(f"[INFO] Skip condition '{name}'")
            continue
        intervals.append({
            "name": name,
            "label": name_to_label[n],
            "start_sec": st,
            "end_sec": ed,
        })
    return intervals


def trim_signals_to_chest(signals, labels):
    chest = signals["chest"]
    wrist = signals["wrist"]

    chest_lens = [len(v) for v in chest.values()]
    n_chest = min(chest_lens)

    chest_trim = {k: np.asarray(v)[:n_chest] for k, v in chest.items()}
    labels_trim = np.asarray(labels)[:n_chest]

    wrist_trim = {}
    for k, v in wrist.items():
        v_arr = np.asarray(v)
        wrist_trim[k] = v_arr[:n_chest] if len(v_arr) >= n_chest else v_arr

    return {"chest": chest_trim, "wrist": wrist_trim}, labels_trim, n_chest


def stats_features_1d(x):
    x = np.asarray(x).astype(np.float32)
    if x.size == 0:
        return np.zeros(7, dtype=np.float32)
    return np.array([
        np.mean(x),
        np.std(x),
        np.min(x),
        np.max(x),
        np.percentile(x, 25),
        np.percentile(x, 50),
        np.percentile(x, 75),
    ], dtype=np.float32)


def extract_handcrafted_features(chest_win, wrist_win):
    feats = []

    def get_channel(win_dict, key, expected_dim):
        if key not in win_dict:
            return None
        x = np.asarray(win_dict[key])
        if x.size == 0:
            return None
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        t, c = x.shape
        if c >= expected_dim:
            return x[:, :expected_dim]
        pad = np.zeros((t, expected_dim - c), dtype=x.dtype)
        return np.concatenate([x, pad], axis=1)

    chest_template = [("ECG", 1), ("EDA", 1), ("Resp", 1), ("Temp", 1), ("ACC", 3)]
    wrist_template = [("BVP", 1), ("EDA", 1), ("TEMP", 1), ("ACC", 3)]

    for key, dim in chest_template:
        ch = get_channel(chest_win, key, dim)
        if ch is None:
            feats.extend([0.0] * (7 * dim))
        else:
            for c in range(dim):
                feats.extend(stats_features_1d(ch[:, c]))

    for key, dim in wrist_template:
        ch = get_channel(wrist_win, key, dim)
        if ch is None:
            feats.extend([0.0] * (7 * dim))
        else:
            for c in range(dim):
                feats.extend(stats_features_1d(ch[:, c]))

    return np.array(feats, dtype=np.float32)


def extract_windows_by_intervals(
    signals,
    labels,
    intervals,
    win_sec: int = WIN_SEC,
    step_sec: int = STEP_SEC,
    fs: int = FS_CHEST_DEFAULT,
):
    chest = signals["chest"]
    wrist = signals["wrist"]

    some_key = next(iter(chest.keys()))
    n = len(chest[some_key])

    win_size = int(win_sec * fs)
    step_size = int(step_sec * fs)

    win_chest_list, win_wrist_list, y_list = [], [], []

    for iv in intervals:
        lab_id = iv["label"]
        st_idx = max(0, int(iv["start_sec"] * fs))
        ed_idx = min(n, int(iv["end_sec"] * fs))

        if ed_idx - st_idx < win_size:
            continue

        cur = st_idx
        while cur + win_size <= ed_idx:
            start = cur
            end = cur + win_size

            seg_labels = labels[start:end]
            main_lab = Counter(seg_labels).most_common(1)[0][0]
            if main_lab != lab_id:
                cur += step_size
                continue

            chest_win = {k: v[start:end] for k, v in chest.items()}
            wrist_win = {}
            for k, v in wrist.items():
                if len(v) >= end:
                    wrist_win[k] = v[start:end]

            win_chest_list.append(chest_win)
            win_wrist_list.append(wrist_win)
            y_list.append(lab_id)

            cur += step_size

    return win_chest_list, win_wrist_list, np.array(y_list, dtype=int)


def build_feature_dataset(subject_ids, data_dir: str = DATA_DIR):
    x_all, y_all, meta_all = [], [], []

    for sid in subject_ids:
        print(f"\nProcessing subject S{sid} ...")
        data = load_subject_pkl(sid, data_dir)
        df_quest = load_subject_quest(sid, data_dir)

        intervals = parse_condition_intervals_from_quest(df_quest)

        signals = data["signal"]
        labels = np.asarray(data["label"])

        fs_est = infer_fs_from_length(labels)
        fs_used = int(round(fs_est))

        signals_trim, labels_trim, _ = trim_signals_to_chest(signals, labels)

        chest_list, wrist_list, y = extract_windows_by_intervals(
            signals_trim, labels_trim, intervals,
            win_sec=WIN_SEC, step_sec=STEP_SEC, fs=fs_used
        )

        local_x = []
        for idx, (cw, ww, lab) in enumerate(zip(chest_list, wrist_list, y)):
            feats = extract_handcrafted_features(cw, ww)
            if feats.size == 0:
                continue
            local_x.append(feats)
            y_all.append(lab)
            meta_all.append({
                "subject": sid,
                "win_id": idx,
                "label": int(lab),
            })

        if len(local_x) > 0:
            x_all.append(np.vstack(local_x))

    if len(x_all) == 0:
        raise RuntimeError("No windows generated for any subject, check pipeline.")

    x_all = np.vstack(x_all)
    y_all = np.array(y_all, dtype=int)
    return x_all, y_all, meta_all