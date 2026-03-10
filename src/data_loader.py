from pathlib import Path

def check_dataset_exists(path: str) -> bool:
    return Path(path).exists()