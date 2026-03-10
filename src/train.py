from data_loader import check_dataset_exists


def main() -> None:
    dataset_path = "data/data.csv"

    print("Starting training pipeline setup...")

    if check_dataset_exists(dataset_path):
        print(f"Dataset found at: {dataset_path}")
    else:
        print(f"Dataset not found: {dataset_path}")


if __name__ == "__main__":
    main()