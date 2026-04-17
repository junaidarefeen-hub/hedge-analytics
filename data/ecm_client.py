"""ECM MCP client for Prismatic gadget rendering and file access.

Handles OAuth credential management and MCP JSON-RPC over SSE transport.
Used by factor_loader.py to fetch factor data from Prismatic gadgets.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import requests
import urllib3

from config import ECM_CREDENTIALS_KEY, ECM_MCP_URL, ECM_REQUEST_TIMEOUT, ECM_TOKEN_BUFFER_SECONDS

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class ECMAuthError(Exception):
    """Raised when ECM authentication fails (missing credentials, expired refresh token)."""


class ECMFetchError(Exception):
    """Raised when an MCP tool call fails."""


# ---------------------------------------------------------------------------
# Credential management
# ---------------------------------------------------------------------------

_CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"
_TOKEN_LOCK = threading.Lock()


def _read_credentials() -> dict:
    """Read ECM OAuth credentials from Claude Code's credential store."""
    try:
        creds = json.loads(_CREDENTIALS_PATH.read_text(encoding="utf-8"))
        return creds["mcpOAuth"][ECM_CREDENTIALS_KEY]
    except FileNotFoundError:
        raise ECMAuthError(
            f"Credentials file not found at {_CREDENTIALS_PATH}. "
            "Ensure Claude Code is authenticated with the ECM MCP server."
        )
    except KeyError:
        raise ECMAuthError(
            f"ECM credentials key '{ECM_CREDENTIALS_KEY}' not found. "
            "Re-authenticate with the ECM MCP server in Claude Code."
        )


def _token_expired(ecm_creds: dict) -> bool:
    """Check if the access token is expired or about to expire."""
    expires_at_ms = ecm_creds.get("expiresAt", 0)
    buffer_ms = ECM_TOKEN_BUFFER_SECONDS * 1000
    return time.time() * 1000 >= expires_at_ms - buffer_ms


def _refresh_token(ecm_creds: dict) -> str:
    """Refresh the access token via Keycloak and write back rotated credentials."""
    discovery = ecm_creds.get("discoveryState", {})
    auth_server = discovery.get("authorizationServerUrl", "")
    if not auth_server:
        raise ECMAuthError("No authorizationServerUrl in credential discoveryState.")

    token_url = f"{auth_server}/protocol/openid-connect/token"
    resp = requests.post(
        token_url,
        data={
            "grant_type": "refresh_token",
            "client_id": ecm_creds["clientId"],
            "refresh_token": ecm_creds["refreshToken"],
            "scope": ecm_creds["scope"],
        },
        verify=False,
        timeout=ECM_REQUEST_TIMEOUT,
    )
    if resp.status_code != 200:
        raise ECMAuthError(
            f"Token refresh failed (HTTP {resp.status_code}). "
            "Re-authenticate with the ECM MCP server in Claude Code."
        )

    tokens = resp.json()
    access_token = tokens["access_token"]

    # Write back rotated credentials
    all_creds = json.loads(_CREDENTIALS_PATH.read_text(encoding="utf-8"))
    ecm_entry = all_creds["mcpOAuth"][ECM_CREDENTIALS_KEY]
    ecm_entry["accessToken"] = access_token
    ecm_entry["refreshToken"] = tokens["refresh_token"]
    ecm_entry["expiresAt"] = int(time.time() * 1000) + tokens.get("expires_in", 60) * 1000
    _CREDENTIALS_PATH.write_text(json.dumps(all_creds, indent=2), encoding="utf-8")

    return access_token


def get_access_token() -> str:
    """Get a valid access token, refreshing if needed. Thread-safe."""
    with _TOKEN_LOCK:
        ecm_creds = _read_credentials()
        if _token_expired(ecm_creds):
            logger.info("ECM access token expired, refreshing...")
            return _refresh_token(ecm_creds)
        return ecm_creds["accessToken"]


# ---------------------------------------------------------------------------
# MCP transport
# ---------------------------------------------------------------------------

_MCP_HEADERS_BASE = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _parse_sse_result(response: requests.Response) -> dict:
    """Parse an SSE response and extract the JSON-RPC result."""
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            payload = json.loads(line[6:])
            if "error" in payload:
                err = payload["error"]
                raise ECMFetchError(f"MCP error {err.get('code')}: {err.get('message')}")
            if "result" in payload:
                return payload["result"]
    raise ECMFetchError("No JSON-RPC result found in SSE response.")


def _mcp_call(access_token: str, method: str, params: dict, msg_id: int = 1) -> dict:
    """Send a JSON-RPC request to the MCP server and return the result."""
    headers = {**_MCP_HEADERS_BASE, "Authorization": f"Bearer {access_token}"}
    body = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}

    resp = requests.post(
        ECM_MCP_URL,
        json=body,
        headers=headers,
        verify=False,
        stream=True,
        timeout=ECM_REQUEST_TIMEOUT,
    )
    if resp.status_code == 401:
        raise ECMAuthError("MCP returned 401 — access token rejected.")
    resp.raise_for_status()

    return _parse_sse_result(resp)


def _mcp_notify(access_token: str, method: str) -> None:
    """Send a JSON-RPC notification (no response expected)."""
    headers = {**_MCP_HEADERS_BASE, "Authorization": f"Bearer {access_token}"}
    body = {"jsonrpc": "2.0", "method": method}
    requests.post(ECM_MCP_URL, json=body, headers=headers, verify=False, timeout=ECM_REQUEST_TIMEOUT)


def _mcp_tool_call(access_token: str, tool_name: str, arguments: dict) -> str:
    """Execute a full MCP handshake and tool call. Returns the tool's text output."""
    # Step 1: initialize
    _mcp_call(access_token, "initialize", {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "hedge-analytics", "version": "1.0.0"},
    }, msg_id=1)

    # Step 2: notify initialized
    _mcp_notify(access_token, "notifications/initialized")

    # Step 3: tool call
    result = _mcp_call(access_token, "tools/call", {
        "name": tool_name,
        "arguments": arguments,
    }, msg_id=2)

    if result.get("isError"):
        content_text = result.get("content", [{}])[0].get("text", "Unknown error")
        raise ECMFetchError(f"Tool '{tool_name}' returned error: {content_text}")

    # Extract text from content blocks
    contents = result.get("content", [])
    for block in contents:
        if block.get("type") == "text":
            return block["text"]

    raise ECMFetchError(f"No text content in tool '{tool_name}' response.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_gadget_csv(
    gadget_id: str,
    parameters: dict | None = None,
) -> str:
    """Render a Prismatic gadget as CSV and return the server file path.

    Args:
        gadget_id: Prismatic gadget ID.
        parameters: Optional form-field overrides for the gadget. Discover
            valid keys via ``ecm_prismatic_get_form_fields``. Common ones
            for plotter-style gadgets: ``plotterStartDate`` and
            ``plotterEndDate`` in ``M/D/YY`` format.

    Raises:
        ECMAuthError: Authentication failed.
        ECMFetchError: Render failed.
    """
    token = get_access_token()
    payload = {"gadget_ids": [gadget_id], "format": "csv"}
    if parameters:
        payload["parameters"] = parameters
    raw = _mcp_tool_call(token, "ecm_prismatic_render_gadgets", payload)
    result = json.loads(raw)
    renders = result.get("renders", [])
    if not renders:
        raise ECMFetchError(f"No render result for gadget {gadget_id}.")
    render = renders[0]
    if render.get("error"):
        raise ECMFetchError(f"Gadget {gadget_id} render error: {render['error']}")
    file_path = render.get("file_path")
    if not file_path:
        raise ECMFetchError(f"No file_path in render result for gadget {gadget_id}.")
    return file_path


def read_share_file(file_path: str) -> str:
    """Read a file from the ECM shared file system.

    Raises:
        ECMAuthError: Authentication failed.
        ECMFetchError: File read failed.
    """
    token = get_access_token()
    return _mcp_tool_call(token, "ecm_get_share_file", {"file_path": file_path})


def fetch_rvx_series(
    ticker: str,
    attribute: str,
    start_date: str,
    end_date: str,
    format: str = "json",
    intraday_mode: str | None = None,
) -> str:
    """Fetch a timeseries from RVX.

    Args:
        ticker: RVX ticker (e.g. "EQSN AAPL", "SPX").
        attribute: RVX attribute (e.g. "EQD(PRICE)").
        start_date: ISO date string (YYYY-MM-DD).
        end_date: ISO date string (YYYY-MM-DD).
        format: "json" for inline data, "csv" for server file path.
        intraday_mode: "hist_live" to include latest live price for current day.

    Returns:
        Text output from the MCP tool (JSON array or CSV file path).

    Raises:
        ECMAuthError: Authentication failed.
        ECMFetchError: Fetch failed.
    """
    token = get_access_token()
    args = {
        "ticker": ticker,
        "attribute": attribute,
        "start_date": start_date,
        "end_date": end_date,
        "format": format,
    }
    if intraday_mode:
        args["intraday_mode"] = intraday_mode
    return _mcp_tool_call(token, "ecm_rvx_fetch_series", args)


def is_available() -> bool:
    """Check if ECM MCP credentials are present and not obviously broken."""
    try:
        _read_credentials()
        return True
    except ECMAuthError:
        return False
