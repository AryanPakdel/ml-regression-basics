from datetime import datetime
from array_utils import rows_to_numpy , compute_basic_stats
from data_loader import check_dataset_path, load_csv_rows, get_csv_summary, find_inconsistent_rows, find_missing_values, \
    find_non_numeric_values
import argparse
import logging
from pathlib import Path
from visualization import save_histogram, save_scatter_plot


def setup_logging():
    logger = logging.getLogger("train_app")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    file_name = log_dir / f"train_{datetime.now():%Y%m%d-%H%M%S}.log"

    file_handler = logging.FileHandler(file_name, mode="w")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def parse_args_from_command_line():
    parser = argparse.ArgumentParser(description="Training pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset"
    )

    return parser.parse_args()


def main() -> None:
    
    args = parse_args_from_command_line()
    logger = setup_logging()
    logger.info("Starting training pipeline")

    #Check if the dataset exist
    dataset_path = Path(args.dataset)
    if not check_dataset_path(dataset_path):
        logger.warning(f"Dataset not found: {dataset_path}")
        return
    logger.info("Dataset path is valid")

    #Loading dataset
    try:
        rows = load_csv_rows(dataset_path)
    except RuntimeError as error:
        logger.error("%s", error)
        return

    if not rows:
        logger.warning("Dataset is empty")
        return

    headers = rows[0]
    data = rows[1:]

    #get dataset summary
    summary = get_csv_summary(headers,data)
    logger.info("CSV loaded successfully")
    logger.info("Row count: %d", summary["row_count"])
    logger.info("Column count: %d", summary["column_count"])
    logger.info("Headers: %s", summary["headers"])

    #Data Validation
    logger.info("Running data quality checks")
    inconsistent_rows = find_inconsistent_rows(headers,data)
    missing_entries = find_missing_values(data)
    non_numeric_values = find_non_numeric_values(data)

    logger.info("Inconsistent rows: %d", len(inconsistent_rows))
    logger.info("Missing values: %d", len(missing_entries))
    logger.info("Non-numeric values: %d", len(non_numeric_values))

    if inconsistent_rows:
        logger.warning("Inconsistent rows found: %s", inconsistent_rows)
    if missing_entries:
        logger.warning("Missing values found: %s", missing_entries)
    if non_numeric_values:
        logger.warning("Non-numeric values found: %s", non_numeric_values)
    if inconsistent_rows or missing_entries or non_numeric_values:
        logger.error("Data validation failed. Stopping pipeline.")
        return

    logger.info("Data validation passed")

    #convert dataset to numpy array
    logger.info("Converting rows to NumPy array")
    array_rows =  rows_to_numpy(data)

    #get dataset basic info
    basic_stats = compute_basic_stats(array_rows)
    logger.info("Basic Dataset Stats")  
    logger.info(f'Dataset shape :  {basic_stats["shape"]}')
    for idx,header in enumerate(headers):
        logger.info(header + " :")
        for key,value in basic_stats.items():
            if key == "shape":
                continue
            logger.info(f"  {key} : {value[idx]}")  
        
    #Visualization
    logger.info("Generating plots")
    for idx, header in enumerate(headers):
        plot_path = save_histogram(array_rows, headers, idx)
        logger.info("Saved histogram for %s at %s", header, plot_path)

    if len(headers) >= 2:
        scatter_path = save_scatter_plot(array_rows, headers, 0, 2)
        logger.info("Saved scatter plot at %s", scatter_path)


if __name__ == "__main__":
    main()