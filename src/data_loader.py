from pathlib import Path
import csv

def check_dataset_path(path :Path) -> bool:
    if not path.exists():
        return False

    if not path.is_file():
        return False

    if path.suffix.lower() != ".csv":
        return False

    return True

def load_csv_rows(path):
    with open(path) as file:
        csv_file = list(csv.reader(file))
    return csv_file    

def get_csv_summary(rows) -> dict:
    if not rows:
        return {
            "row_count": 0,
            "column_count": 0,
            "headers": [],
        }
    
    headers = rows[0]
    data_rows = rows[1:]
    return {
        "row_count": len(data_rows),
        "column_count": len(headers),
        "headers": headers,
    }

        
