from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

RANDOM_STATE = 1

NORM_DF_PATH = "norm_dataframe.csv"
ODDS_DIR = "DATASETS/odds"
ATP_GLOB = "DATASETS/atp_matches_*.csv"
DEFAULT_CACHE_DIR = "DATASETS/checkpoint7_cache"

META_COLS = ["match_idx", "tourney_date", "PLAYER_1", "PLAYER_2"]
TARGET = "RESULT"
TOP_N_SELECTED = 30


@dataclass
class PrematchData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    scaler: StandardScaler
    meta_test: pd.DataFrame
    provenance: dict


def _add_interaction_features(df_feat: pd.DataFrame) -> list:
    df_feat["ELO_RANK_INTERACTION"] = df_feat["ELO_DIFF"] * df_feat["ATP_RANK_DIFF"]
    df_feat["ELO_SURFACE_RANK_RATIO"] = df_feat["ELO_SURFACE_DIFF"] / (
        df_feat["ATP_RANK_DIFF"].abs() + 1e-6
    )
    df_feat["SERVE_COMPOSITE"] = (
        df_feat["P_1STWON_LAST_2000_DIFF"]
        + df_feat["P_2NDWON_LAST_2000_DIFF"]
        + df_feat["P_ACE_LAST_2000_DIFF"]
        - df_feat["P_DF_LAST_2000_DIFF"]
    )
    df_feat["FORM_SHORT"] = df_feat[
        ["ELO_GRAD_5_DIFF", "ELO_GRAD_10_DIFF", "ELO_GRAD_20_DIFF"]
    ].mean(axis=1)
    df_feat["FORM_LONG"] = df_feat[["ELO_GRAD_100_DIFF", "ELO_GRAD_250_DIFF"]].mean(axis=1)
    df_feat["FORM_TREND"] = df_feat["FORM_SHORT"] - df_feat["FORM_LONG"]
    df_feat["H2H_TOTAL"] = df_feat["H2H_DIFF"] + df_feat["H2H_SURFACE_DIFF"]
    return [
        "ELO_RANK_INTERACTION", "ELO_SURFACE_RANK_RATIO", "SERVE_COMPOSITE",
        "FORM_SHORT", "FORM_LONG", "FORM_TREND", "H2H_TOTAL",
    ]


def _add_poly_features(df_feat: pd.DataFrame) -> list:
    top5 = ["ELO_SURFACE_DIFF", "ELO_DIFF", "ATP_RANK_DIFF", "AGE_DIFF", "ATP_POINT_DIFF"]
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_data = poly.fit_transform(df_feat[top5])
    poly_names = list(poly.get_feature_names_out(top5))
    new_poly_names = [n for n in poly_names if n not in top5]
    poly_features = []
    for name in new_poly_names:
        col_name = "POLY_" + name.replace(" ", "_")
        df_feat[col_name] = poly_data[:, poly_names.index(name)]
        poly_features.append(col_name)
    return poly_features


def _add_cluster_features(df_feat: pd.DataFrame, base_features: list,
                          train_idx, test_idx, seed: int) -> list:
    scaler_clust = StandardScaler()
    X_clust_train = scaler_clust.fit_transform(df_feat.loc[train_idx, base_features])
    X_clust_test = scaler_clust.transform(df_feat.loc[test_idx, base_features])

    cluster_features = []
    for k in [5, 10, 15]:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        df_feat[f"CLUSTER_K{k}"] = 0
        df_feat.loc[train_idx, f"CLUSTER_K{k}"] = km.fit_predict(X_clust_train)
        df_feat.loc[test_idx, f"CLUSTER_K{k}"] = km.predict(X_clust_test)
        dist_train = km.transform(X_clust_train)
        dist_test = km.transform(X_clust_test)
        for i in range(k):
            df_feat.loc[train_idx, f"CLUSTER_K{k}_DIST{i}"] = dist_train[:, i]
            df_feat.loc[test_idx, f"CLUSTER_K{k}_DIST{i}"] = dist_test[:, i]
            cluster_features.append(f"CLUSTER_K{k}_DIST{i}")
        cluster_features.append(f"CLUSTER_K{k}")
    return cluster_features


def _surname_odds(name) -> str:
    if pd.isna(name):
        return ""
    parts = str(name).strip().split()
    return parts[0].rstrip(",") if parts else ""


def _surname_atp(name) -> str:
    if pd.isna(name):
        return ""
    parts = str(name).strip().split()
    return parts[-1] if parts else ""


def _load_odds() -> pd.DataFrame:
    frames = []
    for year in range(2000, 2027):
        for path, engine in [(f"{ODDS_DIR}/{year}.xlsx", "openpyxl"),
                             (f"{ODDS_DIR}/{year}.xls", "xlrd")]:
            if not os.path.exists(path):
                continue
            try:
                dfy = pd.read_excel(path, engine=engine)
                if len(dfy) > 0:
                    dfy["year"] = year
                    frames.append(dfy)
                    break
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _add_odds_features(df_feat: pd.DataFrame) -> list:
    odds_df = _load_odds()
    if len(odds_df) == 0 or "Date" not in odds_df.columns:
        df_feat["ODDS_P1"] = np.nan
        df_feat["ODDS_P2"] = np.nan
        df_feat["PROB_DIFF"] = 0.0
        df_feat["LOG_ODDS_RATIO"] = 0.0
        return ["PROB_DIFF", "LOG_ODDS_RATIO"]

    odds_df["Date"] = pd.to_datetime(odds_df["Date"], errors="coerce")
    odds_df = odds_df.dropna(subset=["Date"]).reset_index(drop=True)
    odds_df["P1_surname"] = odds_df["Winner"].apply(_surname_odds).str.lower()
    odds_df["P2_surname"] = odds_df["Loser"].apply(_surname_odds).str.lower()
    for col in ("AvgW", "AvgL"):
        if col not in odds_df.columns:
            odds_df[col] = np.nan

    odds_sorted = odds_df.sort_values("Date", kind="stable").reset_index()
    date_ns = odds_sorted["Date"].values.astype("datetime64[ns]")
    o_s1 = odds_sorted["P1_surname"].to_numpy()
    o_s2 = odds_sorted["P2_surname"].to_numpy()
    o_w = pd.to_numeric(odds_sorted["AvgW"], errors="coerce").to_numpy()
    o_l = pd.to_numeric(odds_sorted["AvgL"], errors="coerce").to_numpy()
    o_orig = odds_sorted["index"].to_numpy()

    window = np.timedelta64(7, "D")
    p1_names = df_feat["PLAYER_1_NAME"].apply(_surname_atp).str.lower().to_numpy()
    p2_names = df_feat["PLAYER_2_NAME"].apply(_surname_atp).str.lower().to_numpy()
    dates = df_feat["tourney_date"].values.astype("datetime64[ns]")

    odds_p1 = np.full(len(df_feat), np.nan)
    odds_p2 = np.full(len(df_feat), np.nan)

    for i in range(len(df_feat)):
        p1, p2, d = p1_names[i], p2_names[i], dates[i]
        if not p1 or not p2 or np.isnat(d):
            continue
        lo = np.searchsorted(date_ns, d - window, side="left")
        hi = np.searchsorted(date_ns, d + window, side="right")
        if hi <= lo:
            continue
        sel = np.argsort(o_orig[lo:hi], kind="stable") + lo  # original concat order
        for j in sel:
            c1, c2 = o_s1[j], o_s2[j]
            if (p1 in c1 or c1 in p1) and (p2 in c2 or c2 in p2):
                odds_p1[i], odds_p2[i] = o_w[j], o_l[j]
                break
            if (p2 in c1 or c1 in p2) and (p1 in c2 or c2 in p1):
                odds_p1[i], odds_p2[i] = o_l[j], o_w[j]
                break

    df_feat["ODDS_P1"] = odds_p1
    df_feat["ODDS_P2"] = odds_p2
    df_feat["PROB_DIFF"] = (1 / df_feat["ODDS_P1"] - 1 / df_feat["ODDS_P2"]).fillna(0)
    df_feat["LOG_ODDS_RATIO"] = (
        np.log(df_feat["ODDS_P1"] / df_feat["ODDS_P2"])
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    return ["PROB_DIFF", "LOG_ODDS_RATIO"]


def _player_id_to_name() -> dict:
    files = glob.glob(ATP_GLOB)
    frames = [
        pd.read_csv(f, usecols=["winner_id", "winner_name", "loser_id", "loser_name"])
        for f in files
    ]
    atp = pd.concat(frames, ignore_index=True)
    w = atp[["winner_id", "winner_name"]].rename(
        columns={"winner_id": "pid", "winner_name": "pname"})
    l = atp[["loser_id", "loser_name"]].rename(
        columns={"loser_id": "pid", "loser_name": "pname"})
    mapping = pd.concat([w, l]).drop_duplicates(subset=["pid"])
    return dict(zip(mapping["pid"], mapping["pname"]))


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prematch_data(seed: int = RANDOM_STATE,
                       cache_dir: str = DEFAULT_CACHE_DIR,
                       force_rebuild: bool = False,
                       verbose: bool = True) -> PrematchData:
    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, "prematch_design.npz")
    meta_path = os.path.join(cache_dir, "prematch_meta_test.parquet")
    prov_path = os.path.join(cache_dir, "prematch_provenance.json")

    if not force_rebuild and all(os.path.exists(p) for p in (npz_path, meta_path, prov_path)):
        if verbose:
            print(f"[prematch_data] loading cache from {cache_dir}")
        z = np.load(npz_path, allow_pickle=True)
        from sklearn.preprocessing import StandardScaler as _SS
        scaler = _SS()
        scaler.mean_ = z["scaler_mean"]
        scaler.scale_ = z["scaler_scale"]
        scaler.var_ = z["scaler_var"]
        scaler.n_features_in_ = z["scaler_mean"].shape[0]
        return PrematchData(
            X_train=z["X_train"], X_test=z["X_test"],
            y_train=z["y_train"], y_test=z["y_test"],
            feature_names=list(z["feature_names"]),
            scaler=scaler,
            meta_test=pd.read_parquet(meta_path),
            provenance=json.load(open(prov_path, encoding="utf-8")),
        )

    t0 = time.time()
    if verbose:
        print("[prematch_data] building dataset (first run is slow: odds matching)...")

    df = pd.read_csv(NORM_DF_PATH, index_col=0)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    base_features = [c for c in df.columns if c not in META_COLS + [TARGET]]

    X_base = df[base_features].copy()
    y = df[TARGET].values
    X_train_base, X_test_base, y_train, y_test = train_test_split(
        X_base, y, test_size=0.2, random_state=seed, stratify=y
    )
    train_idx, test_idx = X_train_base.index, X_test_base.index

    df_feat = df.copy()
    id_to_name = _player_id_to_name()
    df_feat["PLAYER_1_NAME"] = df_feat["PLAYER_1"].map(id_to_name)
    df_feat["PLAYER_2_NAME"] = df_feat["PLAYER_2"].map(id_to_name)

    interaction_features = _add_interaction_features(df_feat)
    poly_features = _add_poly_features(df_feat)
    cluster_features = _add_cluster_features(df_feat, base_features, train_idx, test_idx, seed)
    odds_features = _add_odds_features(df_feat)

    feat_all = sorted(set(
        base_features + interaction_features + poly_features
        + cluster_features + odds_features
    ))

    mi_scores = mutual_info_classif(
        df_feat.loc[train_idx, feat_all], y_train, random_state=seed
    )
    mi_df = pd.DataFrame({"Feature": feat_all, "MI_Score": mi_scores}) \
        .sort_values("MI_Score", ascending=False)
    feat_selected = mi_df.head(TOP_N_SELECTED)["Feature"].tolist()

    X_tr = df_feat.loc[train_idx, feat_selected].values
    X_te = df_feat.loc[test_idx, feat_selected].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_tr)
    X_test = scaler.transform(X_te)

    meta_cols_keep = [
        "PLAYER_1_NAME", "PLAYER_2_NAME", "tourney_date",
        "ELO_DIFF", "ELO_SURFACE_DIFF", "ATP_RANK_DIFF",
        "ODDS_P1", "ODDS_P2", "PROB_DIFF",
    ]
    meta_test = df_feat.loc[test_idx, meta_cols_keep].copy()
    meta_test["RESULT"] = y_test

    provenance = {
        "seed": seed,
        "n_rows": int(len(df)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "norm_dataframe_md5": _file_hash(NORM_DF_PATH),
        "n_base_features": len(base_features),
        "n_feat_all": len(feat_all),
        "n_selected": len(feat_selected),
        "feature_set": "FEAT_SELECTED",
        "odds_coverage_pct": round(
            float(df_feat["ODDS_P1"].notna().mean() * 100), 2),
        "data_source": "norm_dataframe.csv (ATP 1968-2024) + DATASETS/odds + atp_matches",
        "build_seconds": round(time.time() - t0, 1),
    }

    np.savez_compressed(
        npz_path,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=np.array(feat_selected, dtype=object),
        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, scaler_var=scaler.var_,
    )
    meta_test.to_parquet(meta_path)
    json.dump(provenance, open(prov_path, "w", encoding="utf-8"), indent=2)
    if verbose:
        print(f"[prematch_data] built in {provenance['build_seconds']}s, "
              f"selected {len(feat_selected)} features, "
              f"odds coverage {provenance['odds_coverage_pct']}%")

    return PrematchData(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        feature_names=feat_selected, scaler=scaler, meta_test=meta_test,
        provenance=provenance,
    )


if __name__ == "__main__":
    data = load_prematch_data()
    print("FEAT_SELECTED:", data.feature_names)
    print("X_train:", data.X_train.shape, "X_test:", data.X_test.shape)
    print("provenance:", json.dumps(data.provenance, indent=2))
