"""Interactive 15-minute slot simulation for the XBID intraday market.

Loads DAM prices from two sources:
  1. Excel file (Oct-Dec 2025) — scripts/Διπλωματική_-_XBID_Trading.xlsx
  2. HEnEx downloads (Jan 2026 → today) — scripts/dam_cache/

Run from the project root (xbid_hybrid_trader/):

    python scripts/run_simulation.py

Press Enter to advance each 15-minute slot.  Type 'q' + Enter to quit.
At the end of each day the simulation resets automatically with the next
day's DAM prices.
"""

import logging
import sys
from pathlib import Path

# ── make sure scripts/ is importable ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dam_data_manager import DAMDataManager
from historical_data_loader import HistoricalDataLoader
from xbid_trader.market.simulation_session import SimulationSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_simulation")


# ── Engine config (shared across all days) ────────────────────────────────
ENGINE_CONFIG = dict(
    num_products=96,
    event_granularity_seconds=60.0,
    gate_closure_offset_minutes=0.0,
    seed_books_on_init=True,
    initial_half_spread=0.5,
    initial_levels=1,
    initial_level_qty=5.0,
    background_agent_configs={
        "noise_trader": {
            "order_rate_per_minute": 20.0,
            "price_sigma": 1.5,
            "volume_mean": 1.0,
            "volume_std": 0.4,
        },
        "liquidity_provider": {
            "order_rate_per_minute": 8.0,
            "price_offset": 0.5,
            "volume_mean": 3.0,
            "volume_std": 0.3,
        },
        "urgency_trader": {
            "order_rate_per_minute": 6.0,
            "aggression_factor": 8.0,
            "volume_mean": 1.5,
            "volume_std": 0.4,
        },
        "forecast_informed_trader": {
            "order_rate_per_minute": 6.0,
            "signal_sensitivity": 0.1,
            "volume_mean": 1.0,
            "volume_std": 0.2,
        },
    },
)


def main() -> None:
    # ── 1. Load DAM prices ─────────────────────────────────────────────

    manager = DAMDataManager(cache_dir="scripts/dam_cache")

    # Step 1a: Load Oct-Dec 2025 from Excel (HEnEx doesn't serve these)
    excel_path = Path("scripts/Διπλωματική_-_XBID_Trading.xlsx")
    if excel_path.exists():
        print("Loading DAM prices from Excel (Oct–Dec 2025)...")
        excel_loader = HistoricalDataLoader(str(excel_path))
        excel_loader.load()
        manager.load_from_excel(excel_loader, overwrite=False)
        print(f"  → {len(manager)} days loaded from Excel.\n")
    else:
        print(
            "WARNING: Excel file not found at scripts/Διπλωματική_-_XBID_Trading.xlsx\n"
            "         Oct–Dec 2025 prices will be missing.\n"
        )

    # Step 1b: Download Jan 2026 → today from HEnEx
    # (already-loaded Excel dates are skipped automatically)
    print("Checking / downloading DAM price data from HEnEx...")
    manager.fetch_all()

    days = list(manager.iter_days())
    if not days:
        print("ERROR: No DAM price data available.")
        return

    print(f"\nLoaded {len(days)} trading days ({days[0][0]} → {days[-1][0]})\n")

    # ── 2. Run simulation day by day ───────────────────────────────────
    print("╔══════════════════════════════════════════════════════╗")
    print("║   XBID Intraday Market Simulation — Interactive Mode ║")
    print("║   Press Enter to advance 15 min  |  'q' to quit      ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # Initialise with first day's prices
    first_day, first_prices = days[0]
    config = {
        **ENGINE_CONFIG,
        "reference_prices": first_prices,
        "num_products": len(first_prices),   # 92 / 96 / 100 depending on DST
    }
    session = SimulationSession(
        engine_config=config,
        custom_agent=None,
        print_summary=True,
        top_n_levels=5,
    )

    day_index = 0

    while day_index < len(days):
        current_date, ref_prices = days[day_index]

        # Prompt
        try:
            user_input = input(
                f"[{current_date} | Day {day_index + 1}/{len(days)} | "
                f"Slot {session.current_slot:02d}/{len(ref_prices) - 1} | "
                f"T={session.current_slot * 15:04d}min] "
                "Enter to advance (q=quit): "
            )
        except (KeyboardInterrupt, EOFError):
            print("\nSimulation interrupted.")
            break

        if user_input.strip().lower() == "q":
            print("Simulation ended by user.")
            break

        session.step()

        # End of day → reset with next day's prices
        if session.is_day_complete:
            day_index += 1
            if day_index < len(days):
                next_date, next_prices = days[day_index]
                print(f"\n{'=' * 60}")
                print(f"  Day complete: {current_date}")
                print(f"  Next day:     {next_date}")
                print(f"{'=' * 60}\n")
                session.reset(reference_prices=next_prices)
            else:
                print(f"\n{'=' * 60}")
                print("  All available days simulated. Simulation complete.")
                print(f"{'=' * 60}\n")
                break


if __name__ == "__main__":
    main()
