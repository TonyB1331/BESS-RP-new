"""Battery (BESS) asset layer for the XBID intraday RL trader.

Replaces the supplier asset model (DAM load position + exogenous ``Rt``
residual imbalance) with a merchant storage unit that re-optimises a given
day-ahead schedule intraday on XBID, subject to State-of-Charge coupling.

Public API
----------
    BatterySpec               technical parameters + SoC helpers
    dam_arbitrage_schedule    build a feasible day-ahead schedule from DAM prices
    project_feasible          causal SoC clip → (deliverable, soc, imbalance)
    BatteryObservationBuilder 23-feature SoC-aware observation + cash-flow PnL
    BatteryXBIDEnv            Gymnasium environment for the battery agent
"""

from .battery_model import (  # noqa: F401
    BatterySpec,
    dam_arbitrage_schedule,
    project_feasible,
    soc_path,
    is_feasible,
    value_reference,
)
from .battery_observation import (  # noqa: F401
    BatteryObservationBuilder,
    N_FEATURES,
    FEATURE_NAMES,
)
from .battery_env import BatteryXBIDEnv  # noqa: F401
