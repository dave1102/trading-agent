import io
import zipfile

import pytest

from trading_agent.market_data import histdata

CSV_LINE = b"20230102 170000;1.07000;1.07010;1.06990;1.07005;0\n"


class FakeResponse:
    def __init__(self, text="", content=b""):
        self.text = text
        self.content = content

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, page_html, zip_bytes):
        self.page_html = page_html
        self.zip_bytes = zip_bytes
        self.posted = None

    def get(self, url, timeout):
        return FakeResponse(text=self.page_html)

    def post(self, url, data, headers, timeout):
        self.posted = data
        return FakeResponse(content=self.zip_bytes)


def make_zip(name="DAT_ASCII_EURUSD_M1_2023.csv"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, CSV_LINE)
        z.writestr("DAT_ASCII_EURUSD_M1_2023.txt", b"status report")
    return buf.getvalue()


PAGE = '<input type="hidden" name="tk" id="tk" value="abc123" />'


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(histdata.time, "sleep", lambda s: None)


def test_parse_years():
    assert histdata.parse_years("2019-2021") == [2019, 2020, 2021]
    assert histdata.parse_years("2020,2022-2023") == [2020, 2022, 2023]


def test_download_year_extracts_csv(tmp_path):
    session = FakeSession(PAGE, make_zip())
    path = histdata.download_year("eurusd", 2023, out_dir=tmp_path, session=session)
    assert path == tmp_path / "EURUSD" / "DAT_ASCII_EURUSD_M1_2023.csv"
    assert path.read_bytes() == CSV_LINE
    assert session.posted["tk"] == "abc123" and session.posted["fxpair"] == "EURUSD"


def test_existing_file_is_not_downloaded_again(tmp_path):
    existing = tmp_path / "EURUSD" / "DAT_ASCII_EURUSD_M1_2023.csv"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(CSV_LINE)
    session = FakeSession(PAGE, make_zip())
    histdata.download_year("EURUSD", 2023, out_dir=tmp_path, session=session)
    assert session.posted is None


def test_zip_paths_cannot_escape_download_folder(tmp_path):
    session = FakeSession(PAGE, make_zip(name="../../evil.csv"))
    path = histdata.download_year("EURUSD", 2023, out_dir=tmp_path, session=session)
    assert path.parent == tmp_path / "EURUSD"
    assert not (tmp_path.parent / "evil.csv").exists()


def test_missing_form_gives_clear_error(tmp_path):
    session = FakeSession("<html>no form here</html>", b"")
    with pytest.raises(RuntimeError, match="No full-year download"):
        histdata.download_year("EURUSD", 2099, out_dir=tmp_path, session=session)


def test_non_zip_response_gives_clear_error(tmp_path):
    session = FakeSession(PAGE, b"<html>rate limited</html>")
    with pytest.raises(RuntimeError, match="did not return a zip"):
        histdata.download_year("EURUSD", 2023, out_dir=tmp_path, session=session)
