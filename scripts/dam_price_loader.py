"""HEnEx DAM price loader — Greece Mainland 15-min MCP.

Supports two file formats published by HEnEx:

**Raw format** (automatically downloaded via URL):
    One row per asset/MTU combination.  Columns include
    ``BIDDING_ZONE_DESC``, ``SORT`` (1-96) and ``MCP``.
    The MCP is identical across all asset rows for the same MTU.

**Summary format** (manually downloaded pivot file):
    Sheet ``"SPOT_Summary (SELL)"`` with a row labelled
    ``"Greece Mainland  (15min MCP)"`` and 96 values in columns B→CW.

The loader detects the format automatically from the sheet names present
in the workbook.
"""

from __future__ import annotations

import openpyxl
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_greece_mainland_15min_mcp(
    xlsx_path: str,
    sheet_name: str = "SPOT_Summary (SELL)",
) -> np.ndarray:
    """Load the 96 quarter-hour MCP values for Greece Mainland from a HEnEx
    DAM results file.

    Automatically detects whether the file uses the raw data format or the
    summary pivot format and delegates to the appropriate parser.

    Parameters
    ----------
    xlsx_path:
        Path to the xlsx file.
    sheet_name:
        Only used for the *summary* format.  Ignored for raw format files.

    Returns
    -------
    np.ndarray
        Array of shape ``(96,)`` with MCP values in €/MWh, ordered by
        delivery slot (SORT 1 → 96, i.e. 00:00 → 23:45).

    Raises
    ------
    ValueError
        If the expected data cannot be found or the array is incomplete.
    """
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    sheet_names = wb.sheetnames

    # ── Detect format ─────────────────────────────────────────────────
    # Raw format: single sheet, header row contains "BIDDING_ZONE_DESC"
    # Summary format: contains the known pivot sheet name
    if sheet_name in sheet_names:
        return _load_summary_format(wb, sheet_name)
    else:
        return _load_raw_format(wb)


# ---------------------------------------------------------------------------
# Format parsers
# ---------------------------------------------------------------------------

def _load_raw_format(wb: openpyxl.Workbook) -> np.ndarray:
    """Parse the raw row-per-asset format (automatically downloaded files).

    Expects columns (detected by header row):
        BIDDING_ZONE_DESC, SORT, MCP

    Filters rows where BIDDING_ZONE_DESC == "Mainland Greece", then
    collects one MCP value per SORT slot (1-96).
    """
    # Try all sheets — the active sheet may not be the right one
    target_ws = None
    col_zone = col_sort = col_mcp = None
    header_row = None

    for ws in wb.worksheets:
        for row in ws.iter_rows(max_row=20):
            vals = [
                str(c.value).strip() if c.value is not None else ""
                for c in row
            ]
            if "BIDDING_ZONE_DESCR" in vals and "MCP" in vals and "SORT" in vals:
                header_row = row[0].row
                col_zone = vals.index("BIDDING_ZONE_DESCR") + 1  # 1-indexed
                col_sort = vals.index("SORT") + 1
                col_mcp  = vals.index("MCP") + 1
                target_ws = ws
                break
        if target_ws is not None:
            break

    if target_ws is None:
        raise ValueError(
            "Raw format: header row with 'BIDDING_ZONE_DESCR' not found. "
            f"Available sheets: {wb.sheetnames}"
        )

    # ── Collect MCP per SORT slot ─────────────────────────────────────
    mcp_by_slot: dict[int, float] = {}

    for row in target_ws.iter_rows(min_row=header_row + 1, values_only=True):
        zone = row[col_zone - 1]
        sort = row[col_sort - 1]
        mcp  = row[col_mcp  - 1]

        if zone is None:
            continue  # skip empty rows

        zone_str = str(zone).strip()
        # HEnEx uses "Mainland Greece" in the raw format
        if zone_str != "Mainland Greece":
            continue

        if sort is None or mcp is None:
            continue

        slot = int(sort)
        if 1 <= slot <= 96:
            # MCP is identical across all asset rows for the same slot —
            # any value is fine; we just keep the first occurrence.
            if slot not in mcp_by_slot:
                mcp_by_slot[slot] = float(mcp)

    # ── Validate and assemble ─────────────────────────────────────────
    n = len(mcp_by_slot)
    if n not in (92, 96, 100):
        missing = [s for s in range(1, max(mcp_by_slot.keys()) + 1) if s not in mcp_by_slot]
        raise ValueError(
            f"Raw format: unexpected slot count {n} "
            f"(expected 92, 96 or 100). "
            f"Missing SORT values: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    return np.array([mcp_by_slot[s] for s in range(1, n + 1)], dtype=float)


def _load_summary_format(wb: openpyxl.Workbook, sheet_name: str) -> np.ndarray:
    """Parse the summary pivot format (manually downloaded files).

    Looks for the row labelled ``"Greece Mainland  (15min MCP)"`` and
    reads the 96 values from columns B→CW.
    """
    ws = wb[sheet_name]
    label = "Greece Mainland  (15min MCP)"
    row_idx = None

    for row in ws.iter_rows(min_col=1, max_col=1):
        if row[0].value == label:
            row_idx = row[0].row
            break

    if row_idx is None:
        raise ValueError(
            f"Summary format: row '{label}' not found in sheet '{sheet_name}'"
        )

    values = [ws.cell(row_idx, col).value for col in range(2, 98)]
    if len(values) != 96 or any(v is None for v in values):
        raise ValueError("Summary format: expected exactly 96 non-empty MTU values")

    return np.array(values, dtype=float)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    arr = load_greece_mainland_15min_mcp(sys.argv[1])
    np.set_printoptions(suppress=True, linewidth=160)
    print(f"Loaded {len(arr)} slots. Min={arr.min():.2f} Max={arr.max():.2f} Mean={arr.mean():.2f}")
    print(arr)
