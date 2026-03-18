from data_loader import check_dataset_exists
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training pipeline")

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset"
    )

    return parser.parse_args()




def main() -> None:
    
    args = parse_args()

    print("Starting training pipeline")

    if check_dataset_exists(args.data):
        print(f"Dataset found at: {args.data}")
    else:
        print(f"Dataset not found: {args.data}")


if __name__ == "__main__":
    main()