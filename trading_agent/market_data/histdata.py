"""Download free 1-minute forex files from HistData.com.

This submits the same form as the site's own download button, one full year
per request. Files are saved under data/downloads/histdata/<PAIR>/ and are
skipped if already present. The current year is only published month by
month, so only completed years work here.
"""

import io
import re
import time
import zipfile
from pathlib import Path

import requests

from ..config import PROJECT_ROOT

BASE_URL = "https://www.histdata.com"
PAGE_URL = BASE_URL + "/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair}/{year}"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "downloads" / "histdata"
TOKEN_RE = re.compile(r'name="tk" id="tk" value="([^"]+)"')


def parse_years(text: str) -> list[int]:
    """"2019-2021" -> [2019, 2020, 2021]; "2020,2022" -> [2020, 2022]."""
    years = []
    for part in text.split(","):
        if "-" in part:
            first, last = part.split("-")
            years.extend(range(int(first), int(last) + 1))
        else:
            years.append(int(part))
    return years


def download_year(pair: str, year: int, out_dir: Path = DOWNLOAD_DIR, session=None) -> Path:
    """Download one year of M1 data for `pair` (e.g. "EURUSD") and return the CSV path."""
    pair = pair.upper()
    target_dir = Path(out_dir) / pair
    target = target_dir / f"DAT_ASCII_{pair}_M1_{year}.csv"
    if target.exists():
        return target

    session = session or requests.Session()
    page_url = PAGE_URL.format(pair=pair.lower(), year=year)
    page = session.get(page_url, timeout=60)
    page.raise_for_status()
    token = TOKEN_RE.search(page.text)
    if not token:
        raise RuntimeError(f"No full-year download for {pair} {year} (is the year finished?)")

    form = {
        "tk": token.group(1),
        "date": str(year),
        "datemonth": str(year),
        "platform": "ASCII",
        "timeframe": "M1",
        "fxpair": pair,
    }
    response = session.post(BASE_URL + "/get.php", data=form, headers={"Referer": page_url}, timeout=600)
    response.raise_for_status()
    data = io.BytesIO(response.content)
    if not zipfile.is_zipfile(data):
        raise RuntimeError(f"HistData did not return a zip file for {pair} {year}")

    with zipfile.ZipFile(data) as z:
        csv_names = [name for name in z.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"Zip for {pair} {year} contains no CSV file")
        target_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(z.read(csv_names[0]))  # written under our own name, never the zip's path
    time.sleep(1)  # be polite to a free service
    return target
