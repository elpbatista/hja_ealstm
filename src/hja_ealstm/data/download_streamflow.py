import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/streamflow")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Direct CSV download link for HF004 Entity 2 from EDI
URL = (
    "https://portal.edirepository.org/nis/datastream?packageid=knb-lter-and.4341.36&entityid=NEW_ENTITY_ID" # Replace it with the correct URL
)
OUTFILE_CSV = DATA_DIR / "HF004_daily_streamflow_entity2.csv"
OUTFILE_PARQUET = DATA_DIR / "HF004_daily_streamflow_entity2.parquet"


def download_streamflow():
    if OUTFILE_CSV.exists():
        print(f"Already downloaded: {OUTFILE_CSV}")
        return True

    print("Downloading HF004 daily streamflow (Entity 2) from EDI…")
    try:
        r = requests.get(URL)
        r.raise_for_status()
        with open(OUTFILE_CSV, "wb") as f:
            f.write(r.content)
        print(f"Saved to {OUTFILE_CSV}")
        return True
    except Exception as e:
        print(f"[!] Download failed: {e}")
        return False


def process_streamflow():
    if not OUTFILE_CSV.exists():
        print(f"[!] CSV file not found: {OUTFILE_CSV}")
        return None

    print(f"Processing {OUTFILE_CSV}...")
    df = pd.read_csv(OUTFILE_CSV)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Parse date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date")

    # Save to Parquet
    df.to_parquet(OUTFILE_PARQUET)
    print(f"Processed and saved to {OUTFILE_PARQUET}")

    return df


if __name__ == "__main__":
    if download_streamflow():
        process_streamflow()
