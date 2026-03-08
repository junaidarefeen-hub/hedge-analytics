"""Tests for data.fetcher — interval date range validation."""

from datetime import date

from data.fetcher import validate_interval_date_range


class TestValidateIntervalDateRange:
    def test_daily_any_range(self):
        assert validate_interval_date_range("1d", date(2020, 1, 1), date(2023, 1, 1)) is None

    def test_1m_within_limit(self):
        assert validate_interval_date_range("1m", date(2023, 1, 1), date(2023, 1, 5)) is None

    def test_1m_exceeds_limit(self):
        result = validate_interval_date_range("1m", date(2023, 1, 1), date(2023, 2, 1))
        assert result is not None
        assert "7 days" in result

    def test_5m_within_limit(self):
        assert validate_interval_date_range("5m", date(2023, 1, 1), date(2023, 2, 1)) is None

    def test_5m_exceeds_limit(self):
        result = validate_interval_date_range("5m", date(2023, 1, 1), date(2023, 6, 1))
        assert result is not None
        assert "60 days" in result

    def test_1h_within_limit(self):
        assert validate_interval_date_range("1h", date(2022, 1, 1), date(2023, 1, 1)) is None

    def test_1h_exceeds_limit(self):
        result = validate_interval_date_range("1h", date(2020, 1, 1), date(2023, 1, 1))
        assert result is not None
        assert "730 days" in result
