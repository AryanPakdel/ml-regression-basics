from datetime import datetime
from data_loader import check_dataset_exists
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
        "--data",
        type=str,
        required=True,
        help="Path to dataset"
    )

    return parser.parse_args()


def main() -> None:
    
    args = parse_args_from_command_line()
    logger = setup_logging()
    logger.info("Starting training pipeline")

    if check_dataset_exists(args.data):
        logger.info(f"Dataset found at: {args.data}")
    else:
        logger.warning(f"Dataset not found: {args.data}")


if __name__ == "__main__":
    main()