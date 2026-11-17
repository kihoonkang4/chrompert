import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def delete_files_in_dir(directory: Path) -> int:
    count = 0
    if directory.exists():
        for file_path in directory.iterdir():
            if file_path.is_file():
                file_path.unlink()
                count += 1
    return count


def run_reset_bed(data_dir: str = "data") -> None:
    base_data_dir = Path(data_dir)
    removed = sum(
        delete_files_in_dir(base_data_dir / sub)
        for sub in ("bed", "metadata", "log")
    )
    # Remove legacy metadata folder if it exists and is empty
    meta_dir = base_data_dir / "metadata"
    if meta_dir.exists():
        shutil.rmtree(meta_dir, ignore_errors=True)
    logger.info("Cleared %d file(s) under %s/{bed,log}", removed, base_data_dir)
