"""Tests for data.ecm_client — SSE parsing, credential handling, and factor loading."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.ecm_client import (
    ECMAuthError,
    ECMFetchError,
    _parse_sse_result,
    _token_expired,
)


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------


class TestParseSSEResult:
    def test_extracts_result_from_sse(self):
        """Parse a valid SSE response with a JSON-RPC result."""
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            "event: message",
            'data: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26"}}',
        ]
        result = _parse_sse_result(mock_resp)
        assert result["protocolVersion"] == "2025-03-26"

    def test_raises_on_error_response(self):
        """JSON-RPC error in SSE should raise ECMFetchError."""
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            "event: message",
            'data: {"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"Bad request"}}',
        ]
        with pytest.raises(ECMFetchError, match="Bad request"):
            _parse_sse_result(mock_resp)

    def test_raises_on_empty_response(self):
        """No data lines should raise ECMFetchError."""
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = ["", "event: ping"]
        with pytest.raises(ECMFetchError, match="No JSON-RPC result"):
            _parse_sse_result(mock_resp)

    def test_skips_non_data_lines(self):
        """Should ignore SSE lines that aren't 'data:' prefixed."""
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            "event: message",
            ": comment",
            "",
            'data: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}',
        ]
        result = _parse_sse_result(mock_resp)
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Token expiry
# ---------------------------------------------------------------------------


class TestTokenExpired:
    def test_expired_token(self):
        """Token with past expiresAt should be expired."""
        creds = {"expiresAt": 1000}  # way in the past (epoch ms)
        assert _token_expired(creds) is True

    def test_valid_token(self):
        """Token with far-future expiresAt should not be expired."""
        import time
        creds = {"expiresAt": int(time.time() * 1000) + 60_000}
        assert _token_expired(creds) is False

    def test_missing_expires_at(self):
        """Missing expiresAt should be treated as expired."""
        assert _token_expired({}) is True


# ---------------------------------------------------------------------------
# Credential reading
# ---------------------------------------------------------------------------


class TestReadCredentials:
    def test_missing_file_raises(self, tmp_path):
        """Missing credentials file should raise ECMAuthError."""
        with patch("data.ecm_client._CREDENTIALS_PATH", tmp_path / "nonexistent.json"):
            from data.ecm_client import _read_credentials
            with pytest.raises(ECMAuthError, match="not found"):
                _read_credentials()

    def test_missing_key_raises(self, tmp_path):
        """Credentials file without the ECM key should raise ECMAuthError."""
        cred_file = tmp_path / "creds.json"
        cred_file.write_text(json.dumps({"mcpOAuth": {}}))
        with patch("data.ecm_client._CREDENTIALS_PATH", cred_file):
            from data.ecm_client import _read_credentials
            with pytest.raises(ECMAuthError, match="not found"):
                _read_credentials()

    def test_valid_credentials(self, tmp_path):
        """Should return the ECM credential dict when present."""
        cred_file = tmp_path / "creds.json"
        ecm_data = {"accessToken": "tok", "expiresAt": 9999999999999}
        cred_file.write_text(json.dumps({"mcpOAuth": {"ecm|72c3fb7b2b25e5da": ecm_data}}))
        with patch("data.ecm_client._CREDENTIALS_PATH", cred_file):
            from data.ecm_client import _read_credentials
            result = _read_credentials()
            assert result["accessToken"] == "tok"


# ---------------------------------------------------------------------------
# Factor data conversion
# ---------------------------------------------------------------------------


class TestFactorConversion:
    def test_cumulative_to_price_index(self):
        """Cumulative returns starting at 0 should convert to price index starting at 1."""
        cumulative = pd.Series([0.0, -0.01, -0.02, 0.01, 0.03])
        price_index = 1.0 + cumulative
        assert price_index.iloc[0] == 1.0
        assert price_index.iloc[1] == pytest.approx(0.99)
        assert price_index.iloc[4] == pytest.approx(1.03)

    def test_pct_change_recovers_daily_returns(self):
        """pct_change() on the price index should give correct daily returns."""
        cumulative = pd.Series([0.0, 0.01, 0.03, 0.02])
        price_index = 1.0 + cumulative
        daily_returns = price_index.pct_change().dropna()
        # Day 1→2: (1.01 - 1.0) / 1.0 ≈ 0.01
        assert daily_returns.iloc[0] == pytest.approx(0.01, abs=1e-10)
        # Day 2→3: (1.03 - 1.01) / 1.01 ≈ 0.0198
        assert daily_returns.iloc[1] == pytest.approx(0.02 / 1.01, abs=1e-10)


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_returns_false_when_no_creds(self, tmp_path):
        with patch("data.ecm_client._CREDENTIALS_PATH", tmp_path / "nope.json"):
            from data.ecm_client import is_available
            assert is_available() is False

    def test_returns_true_when_creds_present(self, tmp_path):
        cred_file = tmp_path / "creds.json"
        cred_file.write_text(json.dumps({
            "mcpOAuth": {"ecm|72c3fb7b2b25e5da": {"accessToken": "x", "expiresAt": 0}}
        }))
        with patch("data.ecm_client._CREDENTIALS_PATH", cred_file):
            from data.ecm_client import is_available
            assert is_available() is True
