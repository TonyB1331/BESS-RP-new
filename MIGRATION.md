# Migration to MotherDuck (hosted DuckDB)

## Why this changes

The app used to ship a 65 MB `greece_energy_market2.duckdb` file inside the GitHub
repo through **Git LFS**. On Streamlit Community Cloud that is fragile: after the
daily update, the app often received the 130-byte **LFS pointer** instead of the real
binary (and `raw.githubusercontent.com` serves that same pointer, so the auto-heal
downloaded a pointer too), or GitHub's LFS bandwidth quota was exhausted. Result: the
app "stopped seeing" the database until a full reboot re-pulled the file.

**Now the database lives in MotherDuck (hosted DuckDB).** The daily sync writes to the
cloud database; the app reads from it. No file in git, no LFS, no download, no reboot,
no lock fights between the writer and the readers.

Everything goes through one new file, `db_connection.py`. Every reader and the writer
call `db_connection.connect()` instead of `duckdb.connect(<file>)`.

---

## One-time setup

### 1. Create a MotherDuck account + token
- Sign up at motherduck.com (free tier is 10 GB — your DB is ~65 MB).
- Create a **service token**: Settings → Access Tokens → create → copy it.

### 2. Upload your existing database once
You already have the real `greece_energy_market2.duckdb` locally (the 65 MB file — not
the LFS pointer). Upload it to MotherDuck once:

```python
import os, duckdb
os.environ["motherduck_token"] = "PASTE_YOUR_TOKEN"
con = duckdb.connect("md:")
con.execute("CREATE DATABASE greece_energy_market2 FROM 'greece_energy_market2.duckdb'")
print(con.execute("SHOW DATABASES").df())
```

(If a database with that name already exists, drop it first with
`con.execute("DROP DATABASE greece_energy_market2")` or upload under a new name and set
`MD_DATABASE` accordingly.)

### 3. Set the secrets

**Locally** — copy the template and fill it in:
```
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml -> MOTHERDUCK_TOKEN, ENTSOE_API_TOKEN
```

**Streamlit Community Cloud** — App → Settings → **Secrets**, paste:
```
MOTHERDUCK_TOKEN = "..."
ENTSOE_API_TOKEN = "..."
```

The real `secrets.toml` is git-ignored — never commit it.

### 4. Remove the database from git / LFS
The DB is no longer part of the repo. If your existing repo still tracks it via LFS:
```
git rm --cached greece_energy_market2.duckdb
git rm --cached .gitattributes          # LFS tracking removed
git add .gitignore
git commit -m "Move DB to MotherDuck; stop tracking the .duckdb file"
git push
```
(This zip already omits the file, drops `.gitattributes`, and adds a `.gitignore`.)

### 5. Rotate the ENTSO-E token (IMPORTANT)
The old `ENTSOE_API_TOKEN` was hard-coded in `db_sync2.py` and therefore exposed in the
repo. **Revoke it on the ENTSO-E portal and issue a new one**, then put the new value in
your secrets (step 3). The code now reads it from `ENTSOE_API_TOKEN` (env/secrets).

---

## Running the daily sync

`db_sync2.py` now writes straight to MotherDuck. Give it the tokens via env vars:
```
export MOTHERDUCK_TOKEN="..."
export ENTSOE_API_TOKEN="..."
python db_sync2.py
```
No git push, no LFS. The app sees the new data on its next query (see caching note).

---

## Notes

- **Caching / freshness.** The read functions use `@st.cache_data(ttl=1800)`, so results
  are cached for 30 minutes. After a sync the app refreshes within that window; to see
  updates immediately, lower the `ttl` or add a "Refresh data" button that calls
  `st.cache_data.clear()`. This is a *delay*, never the old *error*.
- **XBID optimisation.** The intraday engine (`scripts/`, `xbid_trader/`) has been
  replaced with the latest mature controller (soft-terminal SoC, value decomposition,
  the conservative mode, and the simulator recalibrated on 275 real days). The app's
  `run_daily_sim(...)` call is unchanged — same parameters.
- **What did NOT change.** All SQL, table names, and the dashboard UI are the same; only
  where the connection points changed (local file → `md:greece_energy_market2`).

## Quick test that the fix works
1. Deploy with the secrets set.
2. Run `python db_sync2.py` to write today's data to MotherDuck.
3. Reload the app (no reboot) — it should show the new data. Run the sync again and
   reload: it keeps working, which is the behaviour that used to require a reboot.
