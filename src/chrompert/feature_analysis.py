from concurrent.futures import ThreadPoolExecutor
import gzip
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, roc_curve
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PEACH_COLOR = "#de7e5d"


def _feature_name_from_path(bed_file: Path) -> str:
    """Strip .bed or .bed.gz suffix to obtain the feature name."""
    name = bed_file.name
    if name.endswith(".bed.gz"):
        return name[: -len(".bed.gz")]
    if name.endswith(".bed"):
        return name[: -len(".bed")]
    return name.split(".")[0]


def _read_bed_intervals(bed_file: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read a .bed or .bed.gz file and return per-chromosome sorted (starts, max_ends) arrays."""
    opener = gzip.open if bed_file.name.endswith(".gz") else open
    chrom_starts: dict[str, list[int]] = {}
    chrom_ends: dict[str, list[int]] = {}

    with opener(bed_file, "rt") as f:
        for line in f:
            if not line or line.startswith(("#", "track", "browser")):
                continue
            parts = line.split("\t", 3)
            if len(parts) < 3:
                continue
            chrom = parts[0]
            chrom_starts.setdefault(chrom, []).append(int(parts[1]))
            chrom_ends.setdefault(chrom, []).append(int(parts[2]))

    chrom_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, starts_list in chrom_starts.items():
        starts = np.asarray(starts_list, dtype=np.int64)
        ends = np.asarray(chrom_ends[chrom], dtype=np.int64)
        order = np.argsort(starts, kind="mergesort")
        sorted_starts = starts[order]
        max_ends = np.maximum.accumulate(ends[order])
        chrom_index[chrom] = (sorted_starts, max_ends)

    return chrom_index


def _find_overlapping_peaks_python(
    query_by_chrom: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    bed_file: Path,
) -> tuple[str, set[str]]:
    """Pure-Python/NumPy exact interval intersection without temporary files."""
    feature_name = _feature_name_from_path(bed_file)
    bed_index = _read_bed_intervals(bed_file)
    overlapping: set[str] = set()

    for chrom, (q_starts, q_ends, q_names) in query_by_chrom.items():
        if chrom not in bed_index:
            continue
        b_starts, b_max_ends = bed_index[chrom]
        idx = np.searchsorted(b_starts, q_ends, side="left") - 1
        hit_mask = (idx >= 0) & (b_max_ends[np.maximum(idx, 0)] > q_starts)
        if np.any(hit_mask):
            overlapping.update(q_names[hit_mask].tolist())

    return feature_name, overlapping


def _find_overlapping_peaks_bedtools(
    peaks_bed_stdin: str, bed_file: Path
) -> tuple[str, set[str]]:
    """Run `bedtools intersect -a stdin -b <file> -u` in memory without writing temp files."""
    feature_name = _feature_name_from_path(bed_file)
    result = subprocess.run(
        [
            "bedtools",
            "intersect",
            "-a",
            "stdin",
            "-b",
            str(bed_file),
            "-u",
        ],
        input=peaks_bed_stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    overlapping = {
        line.split("\t", 5)[3] for line in result.stdout.splitlines() if line
    }
    return feature_name, overlapping


def make_feature_matrix(
    bed_list: list[Path],
    feature_matrix_input: Path,
    fc_column: str,
    diff_peaks_column: str,
    output_dir: Path,
    threads: int = 1,
    save_matrix: bool = False,
) -> pd.DataFrame:
    """Overlap ATAC-seq peaks with BED files in memory and return a binary feature matrix."""
    matrix_input_df = pd.read_csv(feature_matrix_input, sep="\t")
    for col in (fc_column, diff_peaks_column):
        if col not in matrix_input_df.columns:
            raise KeyError(
                f"Column '{col}' not found in {feature_matrix_input}. "
                f"Available columns: {list(matrix_input_df.columns)}"
            )

    bed_out_df = matrix_input_df.iloc[:, :3].copy()
    bed_out_df.columns = ["chrom", "start", "end"]
    peak_names = pd.Index(
        [f"Peak_{i}" for i in range(len(bed_out_df))], name="peak_name"
    )
    bed_out_df["name"] = peak_names

    has_bedtools = shutil.which("bedtools") is not None
    logger.info(
        "Intersecting %d peaks with %d BED files in memory using %d thread(s) (%s)...",
        len(bed_out_df),
        len(bed_list),
        threads,
        "bedtools" if has_bedtools else "NumPy interval search",
    )

    if has_bedtools:
        peaks_bed_stdin = bed_out_df.to_csv(sep="\t", header=False, index=False)
        with ThreadPoolExecutor(max_workers=max(1, threads)) as executor:
            results = list(
                executor.map(
                    lambda bf: _find_overlapping_peaks_bedtools(peaks_bed_stdin, bf),
                    bed_list,
                )
            )
    else:
        query_by_chrom: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for chrom, grp in bed_out_df.groupby("chrom", sort=False):
            query_by_chrom[str(chrom)] = (
                grp["start"].to_numpy(dtype=np.int64),
                grp["end"].to_numpy(dtype=np.int64),
                grp["name"].to_numpy(),
            )
        with ThreadPoolExecutor(max_workers=max(1, threads)) as executor:
            results = list(
                executor.map(
                    lambda bf: _find_overlapping_peaks_python(query_by_chrom, bf),
                    bed_list,
                )
            )

    columns_data: dict[str, Any] = {
        fc_column: matrix_input_df[fc_column].to_numpy(),
        diff_peaks_column: matrix_input_df[diff_peaks_column].to_numpy(),
    }
    for feature_name, overlapping_set in results:
        columns_data[feature_name] = peak_names.isin(overlapping_set).astype(int)

    feature_matrix = pd.DataFrame(columns_data, index=peak_names).fillna(0)
    if save_matrix:
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_matrix.to_csv(output_dir / "feature_matrix.csv")

    return feature_matrix


def random_forest(
    feature_matrix: pd.DataFrame,
    target_column: str,
    fold_change: str,
    output_dir: Path,
    threads: int = 1,
) -> tuple[RandomForestClassifier, pd.DataFrame, pd.Series]:
    """Train and tune a Random Forest classifier and save ROC / PR curves."""
    df = feature_matrix.drop(columns=[fold_change], errors="ignore")

    class_counts = df[target_column].value_counts()
    min_class_count = class_counts.min()
    balanced_data = pd.concat(
        [
            df[df[target_column] == cls].sample(min_class_count, random_state=3)
            for cls in class_counts.index
        ]
    )
    balanced_data = balanced_data.sample(frac=1, random_state=3).reset_index(drop=True)

    X = balanced_data.drop(columns=[target_column])
    y = balanced_data[target_column]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=3, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=3, stratify=y_temp
    )

    param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    best_score = -1.0
    best_params: dict[str, Any] = {}
    for params in ParameterGrid(param_grid):
        clf = RandomForestClassifier(random_state=3, n_jobs=threads, **params)
        clf.fit(X_train, y_train)
        val_score = clf.score(X_val, y_val)
        if val_score > best_score:
            best_score = val_score
            best_params = params

    logger.info(
        "Best RF hyperparameters: %s (Validation Accuracy: %.2f)",
        best_params,
        best_score,
    )

    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    rf_classifier = RandomForestClassifier(
        random_state=3, n_jobs=threads, **best_params
    )
    rf_classifier.fit(X_trainval, y_trainval)

    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info("Test Accuracy: %.2f", accuracy)

    y_prob = rf_classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, color="orange", label=f"ROC curve (area = {roc_auc:.2f})")
    ax1.plot(
        [0, 1], [0, 1], color="grey", linestyle="--", alpha=0.5, label="Chance Line (y=x)"
    )
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax1.legend(loc="lower right")

    ax2.plot(
        recall,
        precision,
        color="green",
        label=f"Precision-Recall curve (area = {pr_auc:.2f})",
    )
    ax2.plot(
        [0, 1],
        [0.5, 0.5],
        color="grey",
        linestyle="--",
        alpha=0.5,
        label="Chance Line (y=0.5)",
    )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "roc_pr_curves.pdf", format="pdf")
    plt.close(fig)

    return rf_classifier, X_val, y_val


def random_forest_feature_importance(
    rf_classifier: RandomForestClassifier,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    permutation: bool,
    output_dir: Path,
    threads: int = 1,
) -> pd.DataFrame:
    """Compute MDI (and optionally permutation) feature importances and save plots."""
    features = rf_classifier.feature_names_in_
    mdi_imps = rf_classifier.feature_importances_

    fi_df = pd.DataFrame(
        {"feature": features, "MDI_importance": mdi_imps}
    ).sort_values("MDI_importance", ascending=False, ignore_index=True)

    top30 = fi_df.head(30).copy()
    top30.loc[len(top30)] = ["mean_value", fi_df["MDI_importance"].mean()]

    fig, ax = plt.subplots(figsize=(6, 8))
    y_pos = np.arange(len(top30))
    ax.barh(y_pos, top30["MDI_importance"], color=PEACH_COLOR)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top30["feature"])
    ax.invert_yaxis()
    ax.set_title("MDI Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importances_rf.pdf")
    plt.close(fig)

    if not permutation:
        return fi_df

    with parallel_backend("threading", n_jobs=threads):
        perm = permutation_importance(
            rf_classifier,
            X_eval,
            y_eval,
            n_repeats=10,
            random_state=3,
            n_jobs=threads,
        )

    order = np.argsort(perm.importances_mean)[::-1]
    perm_df = pd.DataFrame(
        {
            "feature": features[order],
            "MDI_importance": mdi_imps[order],
            "permutation_importance": perm.importances_mean[order],
            "permutation_std": perm.importances_std[order],
        }
    )

    top_k = min(30, len(order))
    top_indices = order[:top_k]
    fig, ax = plt.subplots(figsize=(6, 8))
    bp = ax.boxplot(
        [perm.importances[idx] for idx in top_indices[::-1]],
        vert=False,
        labels=features[top_indices[::-1]],
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(PEACH_COLOR)
    ax.set_title("Permutation Importances (Boxplot)")
    ax.set_xlabel("Permutation Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importances_rf_permutation.pdf")
    plt.close(fig)

    return perm_df


def ridge_regression(
    feature_matrix: pd.DataFrame,
    target_column: str,
    change_column: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Fit Ridge regression across 10 random 70% subsamples and plot coefficients."""
    df = (
        feature_matrix.drop(columns=[change_column], errors="ignore")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    X_df = df.drop(columns=[target_column])
    y = df[target_column].to_numpy()
    X = StandardScaler().fit_transform(X_df)

    n_samples = len(df)
    subsample_size = int(round(n_samples * 0.7))
    coef_matrix = np.empty((10, X.shape[1]), dtype=float)

    for i in range(10):
        rng = np.random.default_rng(seed=3 + i)
        idx = rng.choice(n_samples, size=subsample_size, replace=False)
        model = Ridge(alpha=1.0)
        model.fit(X[idx], y[idx])
        coef_matrix[i] = model.coef_

    ridge_df = pd.DataFrame(
        {
            "feature": X_df.columns,
            "Ridge_coef": coef_matrix.mean(axis=0),
            "Ridge_std": coef_matrix.std(axis=0),
        }
    ).sort_values("Ridge_coef", ascending=False, ignore_index=True)

    n_feats = len(ridge_df)
    fig_height = max(6, min(30, int(n_feats * 0.35)))
    fig, ax = plt.subplots(figsize=(6, fig_height))
    vals = ridge_df["Ridge_coef"].to_numpy()[:, None]
    max_abs = max(float(np.abs(vals).max()), 1e-6)
    im = ax.imshow(vals, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_yticks(np.arange(n_feats))
    ax.set_yticklabels(ridge_df["feature"])
    ax.set_xticks([0])
    ax.set_xticklabels(["Ridge_coef"])
    for idx_row, val in enumerate(ridge_df["Ridge_coef"]):
        ax.text(0, idx_row, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Feature Importances from Ridge Regression")
    ax.set_ylabel("Feature")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importances_ridge.pdf", format="pdf")
    plt.close(fig)

    return ridge_df


def remove_features(
    feature_matrix: pd.DataFrame, exclude_features: list[str]
) -> pd.DataFrame:
    """Remove rows overlapping excluded features and drop those feature columns."""
    valid_excludes = [f for f in exclude_features if f in feature_matrix.columns]
    missing = set(exclude_features) - set(valid_excludes)
    for feature in missing:
        logger.warning("Feature '%s' not found in the feature matrix. Skipping.", feature)

    if valid_excludes:
        mask = (feature_matrix[valid_excludes] != 1).all(axis=1)
        feature_matrix = feature_matrix.loc[mask].drop(columns=valid_excludes)

    return feature_matrix.reset_index(drop=True)


def plot_correlation(feature_importance_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot correlation between MDI importance and absolute Ridge coefficient."""
    fig, ax = plt.subplots(figsize=(8, 6))
    abs_ridge = feature_importance_df["Ridge_coef"].abs()
    ax.scatter(
        feature_importance_df["MDI_importance"],
        abs_ridge,
        color=PEACH_COLOR,
        alpha=0.8,
    )

    correlation = feature_importance_df["MDI_importance"].corr(abs_ridge)
    ax.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    ax.set_title("Correlation between MDI Importance and Ridge Coefficient")
    ax.set_xlabel("MDI Importance")
    ax.set_ylabel("Absolute Ridge Coefficient")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_mdi_ridge.pdf", format="pdf")
    plt.close(fig)


def run_feature_analysis(
    input_matrix: str,
    output_dir: str = "data/output",
    bed_dir: str = "data/bed",
    fold_change: str = "log2",
    diff_peaks: str = "change",
    exclude_features: list[str] | None = None,
    permutation: bool = False,
    threads: int = 1,
    save_matrix: bool = False,
) -> None:
    bed_path = Path(bed_dir)
    if not bed_path.exists():
        raise FileNotFoundError(
            f"BED directory '{bed_path}' does not exist. Run 'chrompert frip-filter' first."
        )

    bed_list = sorted(
        p for p in bed_path.iterdir() if p.name.endswith((".bed", ".bed.gz"))
    )
    if not bed_list:
        raise ValueError(f"No .bed or .bed.gz files found in '{bed_path}'.")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exclude_features = exclude_features or []
    if exclude_features:
        logger.info("Excluding features: %s", exclude_features)

    feature_matrix = make_feature_matrix(
        bed_list,
        Path(input_matrix),
        fold_change,
        diff_peaks,
        out_path,
        threads,
        save_matrix=save_matrix,
    )

    if exclude_features:
        feature_matrix = remove_features(feature_matrix, exclude_features)

    rf_classifier, X_val, y_val = random_forest(
        feature_matrix, diff_peaks, fold_change, out_path, threads
    )
    rf_df = random_forest_feature_importance(
        rf_classifier, X_val, y_val, permutation, out_path, threads
    )
    ridge_df = ridge_regression(feature_matrix, fold_change, diff_peaks, out_path)

    combined_df = rf_df.merge(ridge_df, on="feature", how="left")
    plot_correlation(combined_df, out_path)

    out_csv = out_path / "feature_importance.csv"
    combined_df.to_csv(out_csv, index=False)
    logger.info("Saved feature analysis results and plots to %s", out_path)
