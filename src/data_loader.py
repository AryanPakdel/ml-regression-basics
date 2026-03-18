from pathlib import Path

def check_dataset_path(path: Path) -> bool:
    if not path.exists():
        return False

    if not path.is_file():
        return False

    if path.suffix.lower() != ".csv":
        return False

    return True