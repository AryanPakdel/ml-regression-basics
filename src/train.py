from datetime import datetime
from data_loader import check_dataset_path,load_csv_rows,get_csv_summary
import argparse
import logging
from pathlib import Path


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
    dataset_path = Path(args.dataset)
    if not check_dataset_path(dataset_path):
        logger.warning(f"Dataset not found: {dataset_path}")
        return
    
    logger.info("Dataset path is valid")

    try:
        rows = load_csv_rows(dataset_path)
    except RuntimeError as error:
        logger.error("%s", error)
        return

    summary = get_csv_summary(rows)

    logger.info("CSV loaded successfully")
    logger.info("Row count: %d", summary["row_count"])
    logger.info("Column count: %d", summary["column_count"])
    logger.info("Headers: %s", summary["headers"])
        


if __name__ == "__main__":
    main()