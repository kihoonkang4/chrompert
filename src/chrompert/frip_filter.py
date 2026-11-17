from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ENCODE_BASE_URL = "https://www.encodeproject.org"


def _fetch_single_metadata(
    accession: str, session: requests.Session
) -> dict[str, Any] | None:
    """Fetch metadata JSON for a single ENCODE accession in memory."""
    url = f"{ENCODE_BASE_URL}/files/{accession}"
    try:
        resp = session.get(url, headers={"accept": "application/json"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch metadata for %s: %s", accession, exc)
        return None

    metrics = data.get("quality_metrics", [])
    frip = next(
        (m["frip"] for m in metrics if isinstance(m, dict) and "frip" in m), None
    )
    dataset_raw = data.get("dataset", "")
    dataset_parts = dataset_raw.split("/")
    dataset = dataset_parts[2] if len(dataset_parts) > 2 else dataset_raw
    target = data.get("target", {}).get("label")

    return {
        "accession": accession,
        "target": target,
        "dataset": dataset,
        "frip": frip,
    }


def fetch_and_filter_metadata(encode_tsv: Path, parallel: int) -> pd.DataFrame:
    """Fetch ENCODE metadata in memory and keep the highest-FRiP file per target."""
    encode_df = pd.read_csv(encode_tsv, sep="\t", skiprows=1)
    accessions = encode_df["Accession"].dropna().unique().tolist()

    logger.info(
        "Fetching metadata in memory for %d accessions (%d threads)...",
        len(accessions),
        parallel,
    )
    records: list[dict[str, Any]] = []
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [
                executor.submit(_fetch_single_metadata, acc, session)
                for acc in accessions
            ]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    records.append(result)

    if not records:
        raise RuntimeError(f"No valid metadata retrieved from {encode_tsv}.")

    df = pd.DataFrame(records)
    df = (
        df.sort_values(by="frip", ascending=False, na_position="last")
        .drop_duplicates(subset="target", keep="first")
        .reset_index(drop=True)
    )
    return df[["accession", "target", "dataset", "frip"]]


def _download_single_bed(
    target: str,
    accession: str,
    bed_dir: Path,
    min_peaks: int,
    session: requests.Session,
) -> bool:
    """Download a single BED file, checking min_peaks in memory before writing to disk."""
    url = f"{ENCODE_BASE_URL}/files/{accession}/@@download/{accession}.bed.gz"
    out_path = bed_dir / f"{target}.bed.gz"
    try:
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        content = resp.content
    except Exception as exc:
        logger.warning("Failed to download %s (%s): %s", target, accession, exc)
        return False

    if min_peaks > 0:
        line_count = gzip.decompress(content).count(b"\n")
        if line_count < min_peaks:
            logger.info(
                "Skipped %s (%d peaks < %d min_peaks)",
                out_path.name,
                line_count,
                min_peaks,
            )
            return False

    out_path.write_bytes(content)
    logger.info("Downloaded %s", out_path.name)
    return True


def download_bed_files(
    accession_df: pd.DataFrame, parallel: int, min_peaks: int, data_dir: Path
) -> None:
    """Download filtered BED files in parallel and enforce minimum peak counts."""
    bed_dir = data_dir / "bed"
    bed_dir.mkdir(parents=True, exist_ok=True)

    targets_and_accessions = list(
        zip(accession_df["target"], accession_df["accession"])
    )
    logger.info("Downloading %d BED files...", len(targets_and_accessions))

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [
                executor.submit(
                    _download_single_bed, target, accession, bed_dir, min_peaks, session
                )
                for target, accession in targets_and_accessions
            ]
            for future in as_completed(futures):
                future.result()


def write_log(accession_df: pd.DataFrame, data_dir: Path) -> Path:
    """Write TSV log of selected accessions and their FRiP scores."""
    log_dir = data_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.txt"
    idx = 1
    while log_file.exists():
        log_file = log_dir / f"log_{idx}.txt"
        idx += 1

    accession_df.to_csv(log_file, sep="\t", index=False, header=False)
    logger.info("Log file created at %s", log_file)
    return log_file


def run_frip_filter(
    input_file: str,
    min_peaks: int = 0,
    parallel: int = 8,
    data_dir: str = "data",
) -> None:
    data_path = Path(data_dir)
    accession_df = fetch_and_filter_metadata(Path(input_file), parallel)
    download_bed_files(accession_df, parallel, min_peaks, data_path)
    write_log(accession_df, data_path)
