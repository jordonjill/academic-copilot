from __future__ import annotations

import asyncio

from src.infrastructure.memory import stm as stm_module


def test_drain_ltm_tasks_waits_for_pending_tasks():
    async def _case():
        task = asyncio.create_task(asyncio.sleep(0.01))
        stm_module._track_ltm_task(task)
        report = await stm_module.drain_ltm_tasks(timeout_seconds=0.5)
        assert report["initial_pending"] >= 1
        assert report["timed_out"] is False
        assert report["remaining_pending"] == 0
        assert report["cancelled"] == 0

    asyncio.run(_case())


def test_drain_ltm_tasks_cancels_on_timeout():
    async def _case():
        task = asyncio.create_task(asyncio.sleep(1.0))
        stm_module._track_ltm_task(task)
        report = await stm_module.drain_ltm_tasks(timeout_seconds=0.01)
        assert report["initial_pending"] >= 1
        assert report["timed_out"] is True
        assert report["remaining_pending"] == 0
        assert report["cancelled"] >= 1

    asyncio.run(_case())
