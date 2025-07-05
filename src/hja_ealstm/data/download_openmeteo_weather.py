import requests
import pandas as pd
from pathlib import Path

# H.J. Andrews central coordinates
LAT, LON = 44.2, -122.25

START_DATE = "1949-10-01"
END_DATE = "2020-09-30"

OUTDIR = Path("data/meteorology")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "openmeteo_hja_daily.parquet"

URL = "https://archive-api.open-meteo.com/v1/archive"

PARAMS = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "daily": [
        "precipitation_sum",
        "temperature_2m_max",
        "temperature_2m_min",
        "shortwave_radiation_sum",
        "relative_humidity_2m_mean"
    ],
    "timezone": "America/Los_Angeles"
}

def fetch_openmeteo_data():
    print("Requesting data from Open-Meteo API...")
    r = requests.get(URL, params=PARAMS)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df = df.rename(columns={
        "precipitation_sum": "precip_mm",
        "temperature_2m_max": "tmax_c",
        "temperature_2m_min": "tmin_c",
        "shortwave_radiation_sum": "srad_mj_m2",
        "relative_humidity_2m_mean": "rh_mean"
    })

    df.to_parquet(OUTFILE)
    print(f"Saved daily meteorology to {OUTFILE} ({df.shape[0]} records)")

if __name__ == "__main__":
    fetch_openmeteo_data()