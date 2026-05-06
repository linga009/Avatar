# halo_fep/main.py
"""Persistent Mind heartbeat orchestrator.

Runs the subconscious tick loop forever. High free energy triggers wake cycle.
Nightly window triggers LoRA fine-tuning. All external calls (web, LLM) are
isolated so failures degrade gracefully — the heartbeat always continues.

Usage:
    python -m halo_fep.main
"""
from __future__ import annotations

import datetime
import logging
import time
from typing import Any

import jax
import jax.numpy as jnp

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel, halo_fep_step
from halo_fep.memory.schema import Episode
from halo_fep.utils import compute_free_energy

log = logging.getLogger(__name__)


def _is_nightly_window() -> bool:
    """True between 02:00 and 02:15 local time."""
    now = datetime.datetime.now()
    return now.hour == 2 and now.minute < 15


class HeartbeatLoop:
    """Encapsulates one run of the heartbeat loop — injectable for testing."""

    def __init__(
        self,
        cfg: HaloFEPConfig,
        model: HaloFEPModel,
        perception,          # PerceptionPipeline
        memory,              # EpisodeStore
        llm=None,            # LLMBridge (optional; None skips wake cycles)
        goal_updater=None,   # GoalUpdater (optional)
        fep_updater=None,    # FEPUpdater (optional)
        lora_trainer=None,   # LoRATrainer (optional)
        state_compressor=None,  # StateCompressor (optional)
    ) -> None:
        from halo_fep.intellect.state_compressor import StateCompressor
        self.cfg              = cfg
        self.model            = model
        self.carry            = model.init_carry(jax.random.PRNGKey(cfg.seed))
        self.perception       = perception
        self.memory           = memory
        self.llm              = llm
        self.goal_updater     = goal_updater
        self.fep_updater      = fep_updater
        self.lora_trainer     = lora_trainer
        self.state_compressor = state_compressor or StateCompressor(cfg)
        self._prev_fe: float | None = None
        self._nightly_done_date: str | None = None

    def tick(self) -> None:
        """Run one subconscious tick. Never raises — logs errors and returns."""
        # --- Perception ---
        query = self.perception.query_from_beliefs(self.carry)
        try:
            tokens = self.perception.embed(query)
        except Exception as e:
            log.warning(f"Perception failed: {e}. Skipping tick.")
            return

        # --- HALO+FEP step ---
        key, carry_key = jax.random.split(self.carry.key)
        self.carry = self.carry._replace(key=key)
        try:
            self.carry, _ = halo_fep_step(self.model, self.carry, tokens, carry_key)
        except Exception as e:
            log.error(f"halo_fep_step failed: {e}. Skipping tick.")
            return

        fe = float(compute_free_energy(self.carry, self.model))
        if not jnp.isfinite(fe):
            log.error(f"NaN/Inf in free energy — skipping tick.")
            return
        fe_delta = (fe - self._prev_fe) if self._prev_fe is not None else 0.0
        self._prev_fe = fe

        query_embed_np = self.perception.embed_query(query)

        # --- Episode ---
        episode = Episode(
            query=query,
            tokens=jnp.array(tokens).__array__(),
            swarm_mu=jnp.array(self.carry.swarm_mu).__array__(),
            free_energy=fe,
            free_energy_delta=fe_delta,
        )

        # --- FEP matrix update ---
        if self.fep_updater is not None:
            try:
                self.model = self.fep_updater.update(self.model, self.carry, episode)
            except Exception as e:
                log.warning(f"FEP matrix update failed: {e}")

        # --- Goal decay ---
        if self.goal_updater is not None:
            self.model = self.goal_updater.decay(self.model)

        self.memory.add(episode, query_embed=query_embed_np)
        log.info(f"Tick | query={query!r} | FE={fe:.3f} | FE_delta={fe_delta:+.3f}")

        # --- Wake cycle ---
        if fe > self.cfg.wake_threshold and self.llm is not None:
            self._wake_cycle(query, fe, episode)

        # --- Nightly training ---
        today = datetime.date.today().isoformat()
        if _is_nightly_window() and self._nightly_done_date != today:
            self._nightly_learning()
            self._nightly_done_date = today

    def _wake_cycle(self, query: str, fe: float, episode: Episode) -> None:
        log.info(f"Wake cycle triggered (FE={fe:.3f}).")
        from halo_fep.intellect.llm_bridge import parse_llm_output
        try:
            query_embed = self.perception.embed_query(query)
            recent      = self.memory.retrieve(query_embed, k=5)
            prompt      = self.state_compressor.compress(self.carry, recent, query, fe)
            self.llm.load()
            output      = self.llm.think(prompt)
            self.llm.unload()
            log.info(f"Wake output: {output!r}")
            response    = parse_llm_output(output)
            if response.action == "GOAL" and self.goal_updater is not None:
                self.model = self.goal_updater.update_goal(self.model, response.content)
                log.info(f"Goal updated: {response.content!r}")
            elif response.action == "SEARCH":
                log.info(f"New search target: {response.content!r}")
            episode.llm_output = output
            self.memory.update_llm_output(episode.id, output)
        except Exception as e:
            log.error(f"Wake cycle failed: {e}")
            if self.llm is not None:
                self.llm.unload()

    def _nightly_learning(self) -> None:
        log.info("Nightly learning cycle starting.")
        if self.lora_trainer is None:
            return
        try:
            episodes = self.memory.get_high_confidence()
            self.model, info = self.lora_trainer.run(self.model, episodes)
            log.info(f"Nightly learning done: {info}")
        except Exception as e:
            log.error(f"Nightly learning failed: {e}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = HaloFEPConfig(n_tokens=32, wake_threshold=2.5, tick_interval=60)

    # Load or init model
    checkpoint = "data/checkpoints/bootstrap"
    try:
        from halo_fep.training.bootstrap import load_checkpoint
        model = load_checkpoint(cfg, checkpoint)
    except Exception:
        log.info("No checkpoint found — initializing fresh model.")
        model = HaloFEPModel(cfg, jax.random.PRNGKey(cfg.seed))

    from halo_fep.perception.pipeline import PerceptionPipeline
    from halo_fep.memory.episode_store import EpisodeStore
    from halo_fep.intellect.llm_bridge import LLMBridge
    from halo_fep.intellect.goal_updater import GoalUpdater
    from halo_fep.intellect.state_compressor import StateCompressor
    from halo_fep.training.fep_updater import FEPUpdater
    from halo_fep.training.lora_trainer import LoRATrainer

    loop = HeartbeatLoop(
        cfg              = cfg,
        model            = model,
        perception       = PerceptionPipeline(cfg),
        memory           = EpisodeStore("data/episodes/"),
        llm              = LLMBridge(),
        goal_updater     = GoalUpdater(cfg),
        fep_updater      = FEPUpdater(cfg),
        lora_trainer     = LoRATrainer(cfg),
        state_compressor = StateCompressor(cfg),
    )

    log.info("Heartbeat started. Press Ctrl+C to stop.")
    while True:
        tick_start = time.time()
        loop.tick()
        elapsed = time.time() - tick_start
        time.sleep(max(0.0, cfg.tick_interval - elapsed))


if __name__ == "__main__":
    main()
