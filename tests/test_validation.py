"""Tests for utils.validation — parse_tickers, validate_date_range, check_data_sufficiency."""

from datetime import date, timedelta

import pytest

from utils.validation import check_data_sufficiency, parse_tickers, validate_date_range


class TestParseTickers:
    def test_comma_separated(self):
        assert parse_tickers("AAPL, MSFT, GOOGL") == ["AAPL", "MSFT", "GOOGL"]

    def test_space_separated(self):
        assert parse_tickers("AAPL MSFT GOOGL") == ["AAPL", "MSFT", "GOOGL"]

    def test_newline_separated(self):
        assert parse_tickers("AAPL\nMSFT\nGOOGL") == ["AAPL", "MSFT", "GOOGL"]

    def test_mixed_separators(self):
        assert parse_tickers("AAPL, MSFT  GOOGL\nAMZN") == ["AAPL", "MSFT", "GOOGL", "AMZN"]

    def test_deduplication(self):
        assert parse_tickers("AAPL, aapl, MSFT") == ["AAPL", "MSFT"]

    def test_uppercase(self):
        assert parse_tickers("aapl, msft") == ["AAPL", "MSFT"]

    def test_empty_string(self):
        assert parse_tickers("") == []

    def test_whitespace_only(self):
        assert parse_tickers("   ") == []


class TestValidateDateRange:
    def test_valid_range(self):
        start = date(2023, 1, 1)
        end = date(2023, 6, 1)
        assert validate_date_range(start, end) is None

    def test_start_after_end(self):
        msg = validate_date_range(date(2023, 6, 1), date(2023, 1, 1))
        assert msg is not None
        assert "before" in msg.lower()

    def test_same_date(self):
        d = date(2023, 6, 1)
        assert validate_date_range(d, d) is not None

    def test_too_short(self):
        start = date(2023, 6, 1)
        end = start + timedelta(days=20)
        msg = validate_date_range(start, end)
        assert msg is not None
        assert "30" in msg

    def test_exactly_30_days(self):
        start = date(2023, 6, 1)
        end = start + timedelta(days=30)
        assert validate_date_range(start, end) is None


class TestCheckDataSufficiency:
    def test_sufficient(self):
        assert check_data_sufficiency(200, 60) is None

    def test_below_window(self):
        msg = check_data_sufficiency(50, 60)
        assert msg is not None
        assert "Not enough" in msg

    def test_below_2x_window(self):
        msg = check_data_sufficiency(90, 60)
        assert msg is not None
        assert "thin" in msg.lower()

    def test_exactly_2x_window(self):
        assert check_data_sufficiency(120, 60) is None
