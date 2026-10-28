"""Paso 0: extrae solo los CSV del zip de Kaggle (no descomprime los DICOM)."""
import zipfile
from common import ZIP_PATH, META

CSVS = ["train.csv", "train_series.csv", "test.csv", "test_series.csv",
        "sample_submission.csv"]

def main():
    META.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        for name in CSVS:
            (META / name).write_bytes(z.read(name))
            print("extraido", name)

if __name__ == "__main__":
    main()
