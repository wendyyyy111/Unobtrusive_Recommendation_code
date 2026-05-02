# src/attackers.py
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.catalog import item_catalog


def build_attack_dataset(logs, drop_prob: float = 0.3, seed: int = 42):
    num_items = len(item_catalog)
    x_attack, y_attack = [], []

    rng = np.random.RandomState(seed)
    for entry in logs:
        state = entry["state"]
        items = entry["rec_items"]

        items_observed = [it for it in items if rng.rand() > drop_prob]

        feat = np.zeros(num_items, dtype=np.float32)
        for it in items_observed:
            feat[it] += 1.0

        label = 1 if state == 2 else 0
        x_attack.append(feat)
        y_attack.append(label)

    return np.vstack(x_attack), np.array(y_attack, dtype=int)


def build_sequential_attack_dataset(logs, L: int = 5, drop_prob: float = 0.3, seed: int = 42):
    num_items = len(item_catalog)
    rng = np.random.RandomState(seed)

    phi_list, y_list = [], []
    for entry in logs:
        state = entry["state"]
        items = entry["rec_items"]

        items_observed = [it for it in items if rng.rand() > drop_prob]

        feat = np.zeros(num_items, dtype=np.float32)
        for it in items_observed:
            feat[it] += 1.0

        phi_list.append(feat)
        y_list.append(1 if state == 2 else 0)

    phi_arr = np.stack(phi_list, axis=0)
    y_arr = np.array(y_list, dtype=int)

    if len(logs) < L:
        return np.zeros((0, L, num_items), dtype=np.float32), np.zeros((0,), dtype=int)

    x_seq_list, y_seq_list = [], []
    for t in range(L - 1, len(logs)):
        x_seq_list.append(phi_arr[t - L + 1:t + 1])
        y_seq_list.append(y_arr[t])

    return np.stack(x_seq_list, axis=0), np.array(y_seq_list, dtype=int)


class BiLSTMAttacker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_last = h_n[-1]
        logit = self.fc(h_last).squeeze(-1)
        return torch.sigmoid(logit)


def train_bilstm_attacker(
    X_train, y_train, X_val, y_val,
    epochs: int = 30, batch_size: int = 128, lr: float = 1e-3,
    patience: int = 5, device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = BiLSTMAttacker(input_dim=X_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    best_auc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            prob = model(xb)
            loss = criterion(prob, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = model(val_x).detach().cpu().numpy()
        val_auc = roc_auc_score(y_val, val_prob)

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_bilstm_attacker(model, X_test, y_test, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_test, dtype=torch.float32).to(device)
        prob = model(xb).detach().cpu().numpy()
    return roc_auc_score(y_test, prob)


def train_xgb_attacker(
    X_train_seq, y_train, X_val_seq, y_val,
    max_depth: int = 5, n_estimators: int = 500, lr: float = 0.05, early_stopping_rounds: int = 20,
):
    n, l, d = X_train_seq.shape
    x_train = X_train_seq.reshape(n, l * d)
    x_val = X_val_seq.reshape(X_val_seq.shape[0], l * d)

    model = xgb.XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=lr,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    return model


def eval_xgb_attacker(model, X_test_seq, y_test):
    n, l, d = X_test_seq.shape
    x_test = X_test_seq.reshape(n, l * d)
    y_score = model.predict_proba(x_test)[:, 1]
    return roc_auc_score(y_test, y_score)


def train_and_eval_multi_attackers(X, y) -> Dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"LogReg": 0.5, "RandomForest": 0.5, "MLP": 0.5, "BestAUC": 0.5, "BestModel": "None"}

    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    results = {}

    clf_lr = LogisticRegression(max_iter=500, n_jobs=-1)
    clf_lr.fit(x_tr, y_tr)
    results["LogReg"] = roc_auc_score(y_te, clf_lr.predict_proba(x_te)[:, 1])

    clf_rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    clf_rf.fit(x_tr, y_tr)
    results["RandomForest"] = roc_auc_score(y_te, clf_rf.predict_proba(x_te)[:, 1])

    clf_mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.2,
    )
    clf_mlp.fit(x_tr, y_tr)
    results["MLP"] = roc_auc_score(y_te, clf_mlp.predict_proba(x_te)[:, 1])

    best_model = max(["LogReg", "RandomForest", "MLP"], key=lambda k: results[k])
    results["BestAUC"] = results[best_model]
    results["BestModel"] = best_model
    return results


def train_and_eval_strong_attackers(logs, drop_prob: float = 0.3, L: int = 5, seed: int = 42):
    results = {}

    x_att, y_att = build_attack_dataset(logs, drop_prob=drop_prob, seed=seed)
    base_res = train_and_eval_multi_attackers(x_att, y_att)
    results.update(base_res)

    if len(np.unique(y_att)) < 2:
        results["BiLSTM"] = 0.5
        results["XGBoost"] = 0.5
        results["GlobalBestAUC"] = results["BestAUC"]
        results["GlobalBestModel"] = results["BestModel"]
        return results

    x_seq, y_seq = build_sequential_attack_dataset(logs, L=L, drop_prob=drop_prob, seed=seed)
    if x_seq.shape[0] < 10 or len(np.unique(y_seq)) < 2:
        results["BiLSTM"] = 0.5
        results["XGBoost"] = 0.5
        results["GlobalBestAUC"] = results["BestAUC"]
        results["GlobalBestModel"] = results["BestModel"]
        return results

    x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq, test_size=0.3, random_state=seed, stratify=y_seq)
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=seed, stratify=y_train)

    try:
        bilstm = train_bilstm_attacker(x_tr, y_tr, x_val, y_val)
        auc_bilstm = eval_bilstm_attacker(bilstm, x_test, y_test)
    except Exception as e:
        print("[WARN] BiLSTM failed:", e)
        auc_bilstm = 0.5
    results["BiLSTM"] = auc_bilstm

    try:
        xgb_model = train_xgb_attacker(x_tr, y_tr, x_val, y_val)
        auc_xgb = eval_xgb_attacker(xgb_model, x_test, y_test)
    except Exception as e:
        print("[WARN] XGBoost failed:", e)
        auc_xgb = 0.5
    results["XGBoost"] = auc_xgb

    all_models = ["LogReg", "RandomForest", "MLP", "BiLSTM", "XGBoost"]
    best_model = max(all_models, key=lambda k: results.get(k, 0.5))
    results["GlobalBestAUC"] = results[best_model]
    results["GlobalBestModel"] = best_model
    return results


def build_embedding_attack_dataset_from_wesad(X_feat: np.ndarray, y_state: np.ndarray, positive_states=None):
    if positive_states is None:
        positive_states = [2]
    y_sensitive = np.isin(y_state, positive_states).astype(int)
    return X_feat.astype(np.float32), y_sensitive