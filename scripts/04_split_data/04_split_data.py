import pandas as pd
from pathlib import Path
import yaml


def split_data(df: pd.DataFrame):
    """
    Split data into Train, Validation, and Test sets.
    - Zeitbasierter Split über feste Datumsgrenzen aus conf/params.yaml:
      DATA_PREP.TRAIN_DATE, DATA_PREP.VALIDATION_DATE, DATA_PREP.TEST_DATE
    """
    print("✂️ Splitting Data")

    # 1) Sicherstellen, dass wir eine Zeitbasis haben
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    idx = df.index

    # Zeitzone für Datumsvergleich entfernen (falls vorhanden)
    if idx.tz is not None:
        idx_naive = idx.tz_localize(None)
    else:
        idx_naive = idx

    # 2) params.yaml laden
    project_root = Path(__file__).resolve().parents[2]
    params_path = project_root / "conf" / "params.yaml"

    if not params_path.exists():
        raise FileNotFoundError(
            f"conf/params.yaml nicht gefunden unter: {params_path}. "
            "Bitte TRAIN_DATE, VALIDATION_DATE, TEST_DATE dort definieren."
        )

    with open(params_path, "r") as f:
        params = yaml.safe_load(f) or {}

    if "DATA_PREP" not in params:
        raise KeyError(
            "In params.yaml fehlt der Abschnitt 'DATA_PREP'. "
            "Bitte DATA_PREP.TRAIN_DATE / VALIDATION_DATE / TEST_DATE hinzufügen."
        )

    data_prep_cfg = params["DATA_PREP"]

    try:
        train_date_str = data_prep_cfg["TRAIN_DATE"]
        val_date_str = data_prep_cfg["VALIDATION_DATE"]
        test_date_str = data_prep_cfg["TEST_DATE"]
    except KeyError as e:
        raise KeyError(
            f"Fehlender Key in DATA_PREP: {e}. "
            "Benötigt werden TRAIN_DATE, VALIDATION_DATE, TEST_DATE."
        )

    # 3) Strings in Timestamps umwandeln
    train_date = pd.to_datetime(train_date_str)
    val_date = pd.to_datetime(val_date_str)
    test_date = pd.to_datetime(test_date_str)

    print(
        f"   📅 Datumsgrenzen aus params.yaml:"
        f" TRAIN_DATE={train_date.date()},"
        f" VALIDATION_DATE={val_date.date()},"
        f" TEST_DATE={test_date.date()}"
    )

    # 4) Zeitbasierte Masks (wie beim Prof)
    mask_train = idx_naive <= train_date
    mask_val = (idx_naive > train_date) & (idx_naive <= val_date)
    mask_test = (idx_naive > val_date) & (idx_naive <= test_date)

    train = df.loc[mask_train].copy()
    val = df.loc[mask_val].copy()
    test = df.loc[mask_test].copy()

    # 5) Sanity-Check
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            "Mindestens eines der Splits (Train/Val/Test) ist leer.\n"
            "Bitte prüfe TRAIN_DATE / VALIDATION_DATE / TEST_DATE in conf/params.yaml "
            "und ob der Datumsbereich zu deinen Daten passt."
        )

    # 6) Logging
    def _range_info(part, name: str):
        idxp = part.index
        return f"{name}: {len(part):,} rows ({idxp.min().date()} → {idxp.max().date()})"

    print("   Modus: timestamp-based via params.yaml (kein 70/15/15)")
    print("   " + _range_info(train, "Train"))
    print("   " + _range_info(val, "Val"))
    print("   " + _range_info(test, "Test"))

    # 7) Speichern
    processed_path = project_root / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    train.to_parquet(processed_path / "train.parquet")
    val.to_parquet(processed_path / "val.parquet")
    test.to_parquet(processed_path / "test.parquet")

    print(f"✅ Saved splits to {processed_path}")

    return train, val, test
# AM ENDE der Datei hinzufügen:
def main():
    print("=" * 60)
    print("STEP 04: SPLIT DATA")
    print("=" * 60)
    
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
    
    # 1. Load pre-split data from step 03
    input_file = PROCESSED_PATH / "pre_split_data.parquet"
    
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        print("Please run step 03 (Pre Split Preparation) first!")
        return
    
    print(f"📂 Loading pre-split data from: {input_file}")
    df = pd.read_parquet(input_file)
    
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # 2. Apply splitting (rufe die Funktion in der gleichen Datei auf)
    print("\n✂️ Applying temporal split based on params.yaml...")
    train_df, val_df, test_df = split_data(df)  # Aufruf deiner eigenen Funktion!
    
    print("\n✅ Step 04 completed!")
    print("   Files created:")
    print(f"   - {PROCESSED_PATH / 'train.parquet'}")
    print(f"   - {PROCESSED_PATH / 'val.parquet'}")
    print(f"   - {PROCESSED_PATH / 'test.parquet'}")

if __name__ == "__main__":
    main()