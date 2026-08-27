"""Scenario generation for the XBID hybrid trader.

The :mod:`xbid_trader.scenario` package contains stochastic models used to
simulate price processes, forecast errors and imbalance price trajectories.
These models provide the planner and the RL agent with synthetic but
realistically correlated data to condition their decisions.
"""

from .scenario_generator import Scenario, ScenarioGenerator  # noqa: F401