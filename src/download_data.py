"""
Download the Telco Customer Churn dataset from Kaggle using kagglehub.

Run once before training:
    python src/download_data.py
"""

import shutil
from pathlib import Path

import kagglehub

DATASET = "blastchar/telco-customer-churn"
DEST = Path(__file__).resolve().parent.parent / "data"


def main():
    print("Downloading Telco Customer Churn dataset ...")
    path = kagglehub.dataset_download(DATASET)
    src_file = Path(path) / "WA_Fn-UseC_-Telco-Customers-Churn.csv"

    DEST.mkdir(exist_ok=True)
    dest_file = DEST / "telco_churn.csv"
    shutil.copy(src_file, dest_file)
    print(f"Saved to {dest_file}")


if __name__ == "__main__":
    main()
