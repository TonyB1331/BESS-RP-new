"""Abstract interface for custom trading agents.

Any external agent that needs to interact with the simulation should
subclass :class:`CustomAgent` and implement :meth:`on_step_start`.  The
optional :meth:`on_step_end` callback can be used to update internal state
(e.g. reinforcement learning reward signals, position tracking, etc.) after
the background simulation has run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..types import Order
from .market_state import MarketState
from .step_result import StepResult


class CustomAgent(ABC):
    """Abstract base class for agents that interact with the simulation.

    The :class:`~simulation_session.SimulationSession` calls these methods
    once per 15-minute slot:

    1. :meth:`on_step_start` — *before* background agents run.  Any orders
       returned here are injected into the market first, so the agent sees
       the state at the beginning of the slot.
    2. :meth:`on_step_end` — *after* background agents have run and the
       :class:`~step_result.StepResult` is fully populated.

    Parameters
    ----------
    agent_id:
        Unique string identifier embedded in every order this agent creates.
        Must not clash with the background agent ids (``"noise"``, ``"lp"``,
        ``"urgency"``, ``"forecast"``).
    """

    def __init__(self, agent_id: str = "custom_agent") -> None:
        self.agent_id = agent_id

    @abstractmethod
    def on_step_start(self, slot: int, day: int, state: MarketState) -> List[Order]:
        """Generate orders to be submitted at the start of a 15-minute slot.

        This method is called once per step, *before* the background
        simulation runs.  The orders are injected via
        :meth:`~market_engine.MarketEngine.add_external_order` and may
        generate trades immediately if they cross the current book.

        Parameters
        ----------
        slot:
            Current slot index (0–95).
        day:
            Current day counter (starts at 0).
        state:
            Current :class:`~market_state.MarketState` snapshot at the
            beginning of the slot.

        Returns
        -------
        List[Order]
            Orders to submit.  Return an empty list to pass without trading.
            Order ids should be left as ``-1``; the engine assigns them.
        """

    def on_step_end(self, result: StepResult) -> None:
        """React to the completed step.

        Override this to update internal state based on what happened during
        the slot (e.g. PnL tracking, RL reward computation, position
        reconciliation).  The default implementation does nothing.

        Parameters
        ----------
        result:
            The fully populated :class:`~step_result.StepResult` for the
            slot that just completed.
        """
