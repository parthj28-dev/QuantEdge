"""Tests for quantedge.data: parsing, downloading (faked), CSV, and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantedge.data import (
    extract_adjusted_close,
    fetch_prices,
    load_prices_csv,
    parse_tickers,
    save_prices_csv,
    validate_prices,
)
from quantedge.exceptions import DataError, InsufficientDataError


# --------------------------------------------------------------------------- #
# parse_tickers
# --------------------------------------------------------------------------- #
def test_parse_tickers_normalises_and_deduplicates():
    assert parse_tickers(" aapl, msft;RELIANCE.ns  AAPL ^nsei") == ["AAPL", "MSFT", "RELIANCE.NS", "^NSEI"]


def test_parse_tickers_accepts_list_and_special_symbols():
    assert parse_tickers(["brk-b", "M&M.NS", "eurusd=x"]) == ["BRK-B", "M&M.NS", "EURUSD=X"]


@pytest.mark.parametrize("bad", ["", "   ", ",,,"])
def test_parse_tickers_rejects_empty(bad):
    with pytest.raises(DataError, match="No tickers"):
        parse_tickers(bad)


def test_parse_tickers_rejects_invalid_characters():
    with pytest.raises(DataError, match="Invalid ticker"):
        parse_tickers("AAPL, DROP*TABLE")


# --------------------------------------------------------------------------- #
# fetch_prices with an injected downloader (no network)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field_first", [True, False])
def test_fetch_uses_adjusted_close_in_both_column_layouts(
    synthetic_prices, make_yf_frame, fake_downloader, field_first
):
    download = fake_downloader(make_yf_frame(synthetic_prices, field_first=field_first))
    result = fetch_prices("aaa bbb ccc", "2023-01-01", "2024-06-01", downloader=download)

    pd.testing.assert_frame_equal(result.prices, synthetic_prices, check_freq=False)
    assert download.calls == [(["AAA", "BBB", "CCC"], "2023-01-01", "2024-06-01")]
    assert result.report.missing_tickers == ()
    assert result.report.dropped_dates == 0


def test_fetch_single_level_columns_single_ticker(synthetic_prices, fake_downloader):
    raw = pd.DataFrame({"Adj Close": synthetic_prices["AAA"], "Close": synthetic_prices["AAA"] * 2})
    result = fetch_prices("AAA", "2023-01-01", "2024-06-01", downloader=fake_downloader(raw))
    assert list(result.prices.columns) == ["AAA"]
    np.testing.assert_allclose(result.prices["AAA"], synthetic_prices["AAA"])


def test_fetch_reports_ticker_with_no_data(synthetic_prices, make_yf_frame, fake_downloader):
    prices = synthetic_prices.copy()
    prices["BBB"] = np.nan  # yfinance returns an all-NaN column for unknown tickers
    result = fetch_prices("AAA BBB CCC", "2023-01-01", "2024-06-01",
                          downloader=fake_downloader(make_yf_frame(prices)))
    assert result.report.used == ("AAA", "CCC")
    assert result.report.missing_tickers == ("BBB",)
    assert "BBB" not in result.prices.columns


def test_fetch_reports_ticker_absent_from_download(synthetic_prices, make_yf_frame, fake_downloader):
    result = fetch_prices("AAA BBB CCC ZZZ", "2023-01-01", "2024-06-01",
                          downloader=fake_downloader(make_yf_frame(synthetic_prices)))
    assert result.report.missing_tickers == ("ZZZ",)


def test_fetch_all_tickers_missing_raises(synthetic_prices, make_yf_frame, fake_downloader):
    empty = synthetic_prices * np.nan
    with pytest.raises(DataError, match="No usable price data"):
        fetch_prices("AAA BBB CCC", "2023-01-01", "2024-06-01",
                     downloader=fake_downloader(make_yf_frame(empty)))


def test_fetch_empty_download_raises(fake_downloader):
    with pytest.raises(DataError, match="No data returned"):
        fetch_prices("AAA", "2023-01-01", "2024-01-01", downloader=fake_downloader(pd.DataFrame()))


def test_fetch_wraps_network_errors(fake_downloader):
    download = fake_downloader(error=ConnectionError("proxy refused"))
    with pytest.raises(DataError, match="download failed: proxy refused"):
        fetch_prices("AAA", "2023-01-01", "2024-01-01", downloader=download)


def test_fetch_rejects_reversed_dates(fake_downloader):
    with pytest.raises(DataError, match="must be before"):
        fetch_prices("AAA", "2024-01-01", "2023-01-01", downloader=fake_downloader(pd.DataFrame()))


def test_extract_raises_when_adjusted_field_missing(synthetic_prices):
    cols = pd.MultiIndex.from_product([["Close"], synthetic_prices.columns])
    raw = pd.DataFrame(synthetic_prices.to_numpy(), index=synthetic_prices.index, columns=cols)
    with pytest.raises(DataError, match="Adj Close"):
        extract_adjusted_close(raw, ["AAA", "BBB", "CCC"])


# --------------------------------------------------------------------------- #
# validate_prices: missing data is dropped and reported, never filled
# --------------------------------------------------------------------------- #
def test_missing_price_drops_that_date_without_filling(synthetic_prices):
    prices = synthetic_prices.copy()
    gap_date = prices.index[100]
    prices.loc[gap_date, "BBB"] = np.nan

    result = validate_prices(prices)

    assert gap_date not in result.prices.index
    assert result.report.dropped_dates == 1
    assert any("BBB: 1 missing" in note for note in result.report.notes)
    # Every remaining value is exactly the original value: nothing was invented.
    pd.testing.assert_frame_equal(result.prices, synthetic_prices.drop(index=gap_date), check_freq=False)


def test_late_listing_truncates_sample_and_is_reported(synthetic_prices):
    prices = synthetic_prices.copy()
    prices.iloc[:50, prices.columns.get_loc("CCC")] = np.nan  # CCC "lists" on day 51
    result = validate_prices(prices)
    assert result.report.start == synthetic_prices.index[50]
    assert result.report.dropped_dates == 50


@pytest.mark.parametrize("bad_value, message", [(0.0, "non-positive"), (-5.0, "non-positive"),
                                                (np.inf, "infinite")])
def test_invalid_prices_raise_with_ticker_and_date(synthetic_prices, bad_value, message):
    prices = synthetic_prices.copy()
    prices.iloc[10, 1] = bad_value
    with pytest.raises(DataError, match=message) as info:
        validate_prices(prices)
    assert "BBB" in str(info.value)
    assert str(prices.index[10].date()) in str(info.value)


def test_duplicate_dates_raise(synthetic_prices):
    prices = pd.concat([synthetic_prices, synthetic_prices.iloc[[5]]])
    with pytest.raises(DataError, match="Duplicate date"):
        validate_prices(prices)


def test_unsorted_dates_are_sorted(synthetic_prices):
    shuffled = synthetic_prices.sample(frac=1.0, random_state=0)
    result = validate_prices(shuffled)
    assert result.prices.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(result.prices, synthetic_prices, check_freq=False)


def test_non_date_index_raises(synthetic_prices):
    with pytest.raises(DataError, match="dates"):
        validate_prices(synthetic_prices.reset_index(drop=True))


def test_non_numeric_column_raises(synthetic_prices):
    prices = synthetic_prices.copy()
    prices["BBB"] = "n/a"
    with pytest.raises(DataError, match="Non-numeric"):
        validate_prices(prices)


def test_timezone_aware_index_is_made_naive(synthetic_prices):
    prices = synthetic_prices.copy()
    prices.index = prices.index.tz_localize("America/New_York")
    result = validate_prices(prices)
    assert result.prices.index.tz is None


def test_insufficient_data_names_limiting_ticker(synthetic_prices):
    prices = synthetic_prices.copy()
    prices.iloc[:280, 2] = np.nan  # CCC only has 20 prices
    with pytest.raises(InsufficientDataError, match="CCC"):
        validate_prices(prices, min_observations=60)


def test_insufficient_data_short_history(synthetic_prices):
    with pytest.raises(InsufficientDataError, match="Only 9 daily return"):
        validate_prices(synthetic_prices.iloc[:10], min_observations=60)


def test_requested_subset_is_selected_in_order(synthetic_prices):
    result = validate_prices(synthetic_prices, requested=["ccc", "aaa"])
    assert list(result.prices.columns) == ["CCC", "AAA"]


def test_validate_does_not_mutate_input(synthetic_prices):
    original = synthetic_prices.copy()
    prices = synthetic_prices.copy()
    prices.iloc[3, 0] = np.nan
    snapshot = prices.copy()
    validate_prices(prices)
    pd.testing.assert_frame_equal(prices, snapshot)
    assert not original.isna().any().any()


# --------------------------------------------------------------------------- #
# CSV round trip
# --------------------------------------------------------------------------- #
def test_csv_round_trip_is_exact(synthetic_prices, tmp_path):
    path = save_prices_csv(synthetic_prices, tmp_path / "prices.csv")
    loaded = load_prices_csv(path)
    pd.testing.assert_frame_equal(loaded.prices, synthetic_prices, check_freq=False, check_names=False)


def test_csv_blank_cell_is_dropped_not_filled(synthetic_prices, tmp_path):
    prices = synthetic_prices.copy()
    prices.iloc[7, 0] = np.nan
    path = save_prices_csv(prices, tmp_path / "gappy.csv")
    loaded = load_prices_csv(path)
    assert prices.index[7] not in loaded.prices.index
    assert loaded.report.dropped_dates == 1


def test_csv_without_dates_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("name,AAA\nfoo,1\nbar,2\n")
    with pytest.raises(DataError, match="YYYY-MM-DD"):
        load_prices_csv(path)


def test_csv_ambiguous_day_month_dates_are_refused(tmp_path):
    # Is 03-04-2024 the 3rd of April or March 4th? We refuse to guess.
    path = tmp_path / "ambiguous.csv"
    path.write_text("Date,AAA\n03-04-2024,100\n04-04-2024,101\n")
    with pytest.raises(DataError, match="YYYY-MM-DD"):
        load_prices_csv(path, min_observations=1)
