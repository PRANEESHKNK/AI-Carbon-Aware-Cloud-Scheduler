
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

def inspect_dataset(name, file_path):
    file_path = Path(file_path)

    print("\n" + "=" * 60)
    print(f"DATASET: {name}")
    print("=" * 60)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    print("\nShape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nNumerical summary:")
    print(df.describe().round(2))

    print("\nInspection completed.")


def inspect_carbon_data():
    inspect_dataset(
        "Carbon Intensity",
        DATA_DIR / "raw" / "history.csv"
    )


def inspect_resource_data():
    inspect_dataset(
        "Resource Configuration",
        DATA_DIR / "preprocessed" / "region_config.csv"
    )


def inspect_workload_data():
    inspect_dataset(
        "Cloud Workload",
        DATA_DIR / "raw" / "Cell_c.csv"
    )


def inspect_all_datasets():
    inspect_carbon_data()
    inspect_resource_data()
    inspect_workload_data()


if __name__ == "__main__":
    inspect_all_datasets()
