# db_connection.py
# ---------------------------------------------------------------------------
# Single place that opens the database. We now use MotherDuck (hosted DuckDB)
# instead of a local .duckdb file committed to the repo. That removes the whole
# class of "the app stops seeing the DB after the daily update until reboot"
# problems, which were caused by shipping a 65 MB file through Git LFS.
#
# The daily sync (db_sync2.py) WRITES to the same cloud database; the app READS
# from it. There is no file to download, no Git LFS, no reboot needed.
#
# Token resolution order:
#   1. Streamlit secrets  -> st.secrets["MOTHERDUCK_TOKEN"]   (Streamlit Cloud)
#   2. environment var    -> MOTHERDUCK_TOKEN                  (local / cron)
# ---------------------------------------------------------------------------
import os
import duckdb

try:                       # streamlit is present in the app, not in the cron sync
    import streamlit as st
except Exception:          # noqa: BLE001
    st = None

# The MotherDuck database name (create it once — see MIGRATION.md).
MD_DATABASE = os.environ.get("MD_DATABASE", "greece_energy_market2")


def _token() -> str:
    if st is not None:
        try:
            return st.secrets["MOTHERDUCK_TOKEN"]
        except Exception:  # noqa: BLE001
            pass
    return os.environ.get("MOTHERDUCK_TOKEN", "")


def connect(read_only: bool = False):
    """Return a DuckDB connection to the MotherDuck database.

    `read_only` is accepted for call-site compatibility; MotherDuck manages
    read/write concurrency server-side, so readers and the daily writer no
    longer fight over a file lock.
    """
    token = _token()
    if not token:
        raise RuntimeError(
            "MOTHERDUCK_TOKEN is not set. Add it to .streamlit/secrets.toml "
            "(Streamlit Cloud: App settings > Secrets) or export it as an env var."
        )
    # duckdb reads the token from this env var when opening an md: connection
    os.environ["motherduck_token"] = token
    return duckdb.connect(f"md:{MD_DATABASE}")
