from pathlib import Path
import csv


def check_dataset_path(path: Path) -> bool:
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
    headers = rows[0]
    data_rows = rows[1:]
    return {
        "row_count": len(data_rows),
        "column_count": len(headers),
        "headers": headers,
    }


def find_inconsistent_rows(rows):
    inconsistent_rows = []
    headers = rows[0]
    expected_length = len(headers)
    for idx, row in enumerate(rows):
        if len(row) != expected_length:
            inconsistent_rows.append((idx, row))

    return inconsistent_rows


def find_missing_values(rows):
    missing_entries = []
    for row_idx, row in enumerate(rows):
        if row_idx == 0:
            continue
        for col_idx, value in enumerate(row):
            if value is None or value.strip() == "":
                missing_entries.append((row_idx, col_idx))

    return missing_entries


def find_non_numeric_values(rows):
    non_numeric_values = []
    for row_idx, row in enumerate(rows):
        if row_idx == 0:
            continue
        for col_idx, value in enumerate(row):
            try:
                float(value)
            except (ValueError, TypeError):
                non_numeric_values.append((row_idx, col_idx, value))

    return non_numeric_values

