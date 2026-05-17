# halo_fep/main.py
"""Persistent Mind heartbeat orchestrator.

Runs the subconscious tick loop indefinitely.  High free energy triggers the
LLM wake cycle.  The nightly window triggers LoRA fine-tuning.  All external
calls (web, LLM) are isolated so failures degrade gracefully — the heartbeat
always continues.

Fixes applied
-------------
* **Nightly guard (Design Issue 5)**: ``_nightly_done_date`` is only set after
  *successful* nightly training.  Previously it was set unconditionally, so a
  crashed dream would silently skip training for the rest of the night.

* **FEPUpdater now receives real soft_obs**: ``halo_fep_step`` returns
  ``soft_obs`` as the second element of the output tuple.  We now forward it
  to ``FEPUpdater.update()`` instead of letting the updater approximate it
  from the action distribution.

Usage
-----
    python -m halo_fep.main
"""
from __future__ import annotations

import datetime
import logging
import signal
import time
from typing import Any

import jax
import jax.numpy as jnp

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel, halo_fep_step
from halo_fep.memory.schema import Episode
from halo_fep.utils import compute_free_energy
from halo_fep.paths import ensure_dirs, EPISODE_DIR, BOOTSTRAP_CKPT

log = logging.getLogger(__name__)


def _is_nightly_window() -> bool:
    """Return True between 02:00 and 02:15 local time."""
    now = datetime.datetime.now()
    return now.hour == 2 and now.minute < 15


class HeartbeatLoop:
    """Encapsulates one run of the subconscious heartbeat loop.

    All external dependencies (perception, memory, LLM, training) are injected
    so the loop can be fully tested with mocks.

    Parameters
    ----------
    cfg              : System configuration.
    model            : Initialised HaloFEPModel.
    perception       : PerceptionPipeline — web fetch + embedding.
    memory           : EpisodeStore — SQLite + FAISS memory.
    llm              : LLMBridge — optional; None disables wake cycles.
    goal_updater     : GoalUpdater — optional.
    fep_updater      : FEPUpdater — optional.
    lora_trainer     : LoRATrainer — optional; None disables nightly dreaming.
    state_compressor : StateCompressor — optional; default instance used if None.
    """

    def __init__(
        self,
        cfg: HaloFEPConfig,
        model: HaloFEPModel,
        perception,
        memory,
        llm=None,
        goal_updater=None,
        fep_updater=None,
        lora_trainer=None,
        state_compressor=None,
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
        self._prev_fe: float | None       = None
        self._nightly_done_date: str | None = None

    def tick(self) -> None:
        """Run one subconscious tick.  Never raises — logs errors and returns.

        Tick sequence:
        1. Perception (web fetch + embed → token tensor)
        2. HALO+FEP step (update carry)
        3. Free-energy computation
        4. Episode persistence
        5. FEP matrix update (using real soft_obs)
        6. Goal decay
        7. Wake cycle if FE > threshold
        8. Nightly learning if in time window
        """
        # --- 1. Perception ---
        query = self.perception.query_from_beliefs(self.carry)
        try:
            tokens = self.perception.embed(query)
        except Exception as e:
            log.warning(f"Perception failed: {e}. Skipping tick.")
            return

        # --- 2. HALO+FEP step ---
        key, carry_key = jax.random.split(self.carry.key)
        self.carry = self.carry._replace(key=key)
        try:
            self.carry, (h_out, soft_obs, v_pred, v_target) = halo_fep_step(
                self.model, self.carry, tokens, carry_key
            )
        except Exception as e:
            log.error(f"halo_fep_step failed: {e}. Skipping tick.")
            return

        # --- 3. Free-energy ---
        fe = float(compute_free_energy(self.carry, self.model))
        if not jnp.isfinite(fe):
            log.error("NaN/Inf in free energy — skipping tick.")
            return
        fe_delta     = (fe - self._prev_fe) if self._prev_fe is not None else 0.0
        self._prev_fe = fe

        # --- 4. Episode ---
        query_embed_np = self.perception.embed_query(query)
        episode = Episode(
            query             = query,
            tokens            = jnp.array(tokens).__array__(),
            swarm_mu          = jnp.array(self.carry.swarm_mu).__array__(),
            free_energy       = fe,
            free_energy_delta = fe_delta,
        )

        # --- 5. FEP matrix update (using REAL soft_obs from ObsBridge) ---
        if self.fep_updater is not None:
            try:
                self.model = self.fep_updater.update(
                    self.model, self.carry, episode, soft_obs
                )
            except Exception as e:
                log.warning(f"FEP matrix update failed: {e}")

        # --- 6. Goal decay ---
        if self.goal_updater is not None:
            self.model = self.goal_updater.decay(self.model)

        # Persist episode (after FEP update so model state is consistent)
        self.memory.add(episode, query_embed=query_embed_np)
        log.info(f"Tick | query={query!r} | FE={fe:.3f} | FE_delta={fe_delta:+.3f}")

        # --- 7. Wake cycle ---
        if fe > self.cfg.wake_threshold and self.llm is not None:
            self._wake_cycle(query, fe, episode)

        # --- 8. Nightly learning ---
        today = datetime.date.today().isoformat()
        if _is_nightly_window() and self._nightly_done_date != today:
            success = self._nightly_learning()
            # KEY FIX: Only mark today as done if training succeeded.
            # Previously this was set unconditionally, silently skipping
            # the rest of the night if training crashed.
            if success:
                self._nightly_done_date = today

    def _wake_cycle(self, query: str, fe: float, episode: Episode) -> None:
        """Invoke the LLM for deep reasoning when free energy is high.

        Always unloads the LLM — even on error — to free CUDA memory.

        Parameters
        ----------
        query   : Current search query (used for memory retrieval).
        fe      : Current free-energy value.
        episode : Current episode (updated with LLM output if successful).
        """
        log.info(f"Wake cycle triggered (FE={fe:.3f}).")
        from halo_fep.intellect.llm_bridge import parse_llm_output
        try:
            query_embed = self.perception.embed_query(query)
            recent      = self.memory.retrieve(query_embed, k=5)
            prompt      = self.state_compressor.compress(self.carry, recent, query, fe)
            self.llm.load()
            output      = self.llm.think(prompt)
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
        finally:
            # Always unload — even if generation raised an exception
            if self.llm is not None:
                self.llm.unload()

    def _nightly_learning(self) -> bool:
        """Run the nightly LoRA fine-tuning cycle.

        Returns
        -------
        True if training completed successfully (even if it was reverted),
        False if training raised an unhandled exception.
        """
        log.info("Nightly learning cycle starting.")
        if self.lora_trainer is None:
            return True  # nothing to do; not a failure
        try:
            episodes = self.memory.get_high_confidence()
            if not episodes:
                log.info("No high-confidence episodes for nightly training.")
                return True
            self.model, info = self.lora_trainer.run(self.model, episodes)
            log.info(f"Nightly learning done: {info}")
            # Flush FAISS to disk after successful training
            self.memory.flush()
            return True
        except Exception as e:
            log.error(f"Nightly learning failed: {e}")
            return False


def main() -> None:
    """Entry point: configure logging, build all components, run heartbeat."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Ensure all data directories exist before any component tries to open files
    ensure_dirs()

    cfg = HaloFEPConfig(wake_threshold=2.5, tick_interval=60)
    # n_tokens defaults to 32 (fixed in config.py); no need to pass explicitly.

    # --- Model: load checkpoint or init fresh ---
    try:
        from halo_fep.training.bootstrap import load_checkpoint
        model = load_checkpoint(cfg, BOOTSTRAP_CKPT)
    except Exception:
        log.info("No checkpoint found — initialising fresh model.")
        model = HaloFEPModel(cfg, jax.random.PRNGKey(cfg.seed))

    from halo_fep.perception.pipeline    import PerceptionPipeline
    from halo_fep.memory.episode_store   import EpisodeStore
    from halo_fep.intellect.llm_bridge   import LLMBridge
    from halo_fep.intellect.goal_updater import GoalUpdater
    from halo_fep.intellect.state_compressor import StateCompressor
    from halo_fep.training.fep_updater   import FEPUpdater
    from halo_fep.training.lora_trainer  import LoRATrainer

    memory = EpisodeStore(str(EPISODE_DIR))

    loop = HeartbeatLoop(
        cfg              = cfg,
        model            = model,
        perception       = PerceptionPipeline(cfg),
        memory           = memory,
        llm              = LLMBridge(),
        goal_updater     = GoalUpdater(cfg),
        fep_updater      = FEPUpdater(cfg),
        lora_trainer     = LoRATrainer(cfg),
        state_compressor = StateCompressor(cfg),
    )

    # --- Graceful shutdown on SIGINT / SIGTERM ---
    shutdown_requested = False

    def _handle_signal(sig, frame):
        nonlocal shutdown_requested
        log.info(f"Received signal {sig} — initiating graceful shutdown.")
        shutdown_requested = True

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("Heartbeat started. Press Ctrl+C to stop.")
    while not shutdown_requested:
        tick_start = time.time()
        loop.tick()
        elapsed    = time.time() - tick_start
        sleep_time = max(0.0, cfg.tick_interval - elapsed)
        if elapsed > 1.5 * cfg.tick_interval:
            log.warning(
                f"Tick overrun: {elapsed:.2f}s (threshold {1.5 * cfg.tick_interval:.2f}s)"
            )
        time.sleep(sleep_time)

    # Flush FAISS on shutdown so no inserts since last periodic write are lost
    memory.flush()
    log.info("Heartbeat loop exited cleanly.")


if __name__ == "__main__":
    main()
