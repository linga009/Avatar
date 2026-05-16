"""Organism — the unified psyche that lives in the physics body.

Integrates drives, emotions, self-model, circadian rhythm, and
prefrontal cortex (Qwen3 0.6B via Ollama/LoRA) into a single coherent
agent that modulates the physics engine.

v3.1 fixes:
  - Zero-result tracking: consecutive_zero_results drives frustration + forced escape
  - Frustration handling: overrides both PFC and emotion system for emergency topic change
  - Semantic dedup: passes recent queries and dead queries to PFC
  - Starvation emergency: bypasses all query generation when info-starved
  - Post-dream exploration plan: PFC generates topics before waking
  - Meta-cognitive query tracking: records what works and what doesn't
"""
from __future__ import annotations
import logging
from collections import deque
from halo3.psyche.drives import DriveState
from halo3.psyche.emotions import EmotionState
from halo3.psyche.self_model import SelfModel
from halo3.psyche.circadian import CircadianClock
from halo3.psyche.prefrontal import PrefrontalCortex

log = logging.getLogger(__name__)


class Organism:
    """The living psyche that inhabits the HoloBiont physics body."""

    def __init__(self, seed_topics: list[str]) -> None:
        self.drives = DriveState()
        self.emotions = EmotionState()
        self.self_model = SelfModel.load()
        self.clock = CircadianClock()
        self.prefrontal = PrefrontalCortex()
        self.seed_topics = seed_topics
        self.current_topic_idx = 0
        self.current_query: str = seed_topics[0] if seed_topics else "research"
        self._exploit_streak = 0
        self._consecutive_zero_results = 0
        self._recent_queries: deque = deque(maxlen=20)
        self._prev_query: str = ""
        self._exploration_plan: list[str] = []  # post-dream topics to explore

    def tick(
        self,
        r_mean: float,
        fe_delta: float,
        texts: list[str],
        current_query: str,
    ) -> dict:
        """Process one tick of lived experience."""

        perception_failed = len(texts) == 0

        # Track zero-result streak
        if perception_failed:
            self._consecutive_zero_results += 1
        else:
            # Record dead query if it failed many times then we moved on
            if self._consecutive_zero_results >= 3 and self._prev_query:
                self.self_model.record_dead_query(self._prev_query)
            self._consecutive_zero_results = 0

        # Track query changes for novelty drive
        topic_changed = current_query != self._prev_query
        self._prev_query = current_query

        # 1. Update drives (with perception failure signal)
        self.drives.update(
            r_mean, fe_delta,
            perception_failed=perception_failed,
            topic_changed=topic_changed,
        )

        # 2. Compute emotion (with failure context)
        emotion, intensity = self.emotions.update(
            r_mean, fe_delta,
            perception_failed=perception_failed,
            consecutive_failures=self._consecutive_zero_results,
        )

        # 3. Determine finding
        finding = None
        if r_mean > 0.6 and texts:
            pfc_finding = self.prefrontal.interpret_finding(texts, current_query, r_mean)
            finding = pfc_finding or f"{'; '.join(texts[:3])}"

        # 4. Update self-model
        topic_key = self._extract_topic(current_query)
        self.self_model.update(topic_key, r_mean, emotion, finding)

        # 5. Record whether PFC's last query worked
        self.prefrontal.record_query_result(had_results=not perception_failed)

        # 6. Decide next query — with layered fallbacks
        next_query = self._decide_query(emotion, r_mean, current_query, texts)

        # Track
        self._recent_queries.append(next_query)

        # 7. Modulate physics
        coupling_mod = self.clock.modulate_coupling(1.0, self.drives.fatigue)
        if self.drives.is_satiated:
            coupling_mod *= (1.0 - self.drives.satiation * 0.8)

        # 8. Build log line
        emo_emoji = self.emotions.emoji()
        drives_str = self.drives.summary()
        log_line = (
            f"{emo_emoji} {emotion:12s} (i={intensity:.2f}) | "
            f"{drives_str}"
        )

        return {
            "emotion": emotion,
            "intensity": intensity,
            "next_query": next_query,
            "coupling_mod": coupling_mod,
            "finding": finding,
            "log_line": log_line,
            "needs_dream": self.drives.needs_dream or self.clock.should_dream_today,
            "perception_failed": perception_failed,
        }

    def _decide_query(
        self, emotion: str, r_mean: float, current_query: str, texts: list[str]
    ) -> str:
        """Layered query decision with emergency overrides.

        Priority order:
        1. Starvation emergency → random seed topic (bypass everything)
        2. Frustration → drastic topic change (bypass PFC)
        3. Post-dream exploration plan (if available)
        4. PFC generation (with dedup + dead-query checks)
        5. Emotion-based fallback
        """
        # --- Layer 1: Starvation emergency ---
        if self.drives.is_information_starved:
            new_topic = self._random_seed_topic()
            log.warning(f"STARVATION OVERRIDE: forcing topic change to '{new_topic}'")
            self._consecutive_zero_results = 0  # reset after override
            return new_topic

        # --- Layer 2: Frustration override ---
        if emotion == "frustration":
            # Don't ask the PFC — it might be part of the problem
            new_topic = self._escape_dead_end(current_query)
            log.info(f"FRUSTRATION: escaping dead-end, switching to '{new_topic}'")
            return new_topic

        # --- Layer 3: Post-dream exploration plan ---
        if self._exploration_plan:
            planned = self._exploration_plan.pop(0)
            log.info(f"Following exploration plan: '{planned}'")
            return planned

        # --- Layer 4: PFC generation ---
        pfc_query = self.prefrontal.generate_query(
            current_query, emotion, r_mean, texts,
            self.self_model.strengths,
            consecutive_failures=self._consecutive_zero_results,
            dead_queries=self.self_model.dead_queries,
        )
        if pfc_query:
            log.debug(f"Prefrontal: generated query '{pfc_query}'")
            return pfc_query

        # --- Layer 5: Emotion-based fallback ---
        return self._emotion_query(emotion, r_mean, current_query, texts)

    def _emotion_query(
        self, emotion: str, r_mean: float, current_query: str, texts: list[str]
    ) -> str:
        """Emotion-driven query selection — the organism's instinct."""

        if emotion == "satisfaction":
            self._exploit_streak += 1
            if self._exploit_streak > 5:
                return self._next_seed_topic()
            return current_query

        elif emotion == "pride":
            self._exploit_streak += 1
            if texts:
                refinement = " ".join(texts[0].split()[:4])
                return current_query + " " + refinement
            return current_query

        elif emotion == "curiosity":
            self._exploit_streak = 0
            # Don't blindly return current_query if it's been failing
            if self._consecutive_zero_results >= 2:
                return self._next_seed_topic()
            return current_query

        elif emotion == "boredom":
            self._exploit_streak = 0
            if self.self_model.weaknesses:
                return self.self_model.weaknesses[0]
            return self._random_seed_topic()

        elif emotion == "anxiety":
            self._exploit_streak = 0
            if self.self_model.strengths:
                return self.self_model.strengths[0]
            return self._next_seed_topic()

        return self._next_seed_topic()

    def _escape_dead_end(self, current_query: str) -> str:
        """Find a topic maximally different from the current dead-end."""
        # Try weaknesses first (unexplored territory)
        if self.self_model.weaknesses:
            return self.self_model.weaknesses[0]

        # Try seed topic most different from current query
        best_topic = None
        best_diff = -1.0
        current_words = set(current_query.lower().split())
        for topic in self.seed_topics:
            topic_words = set(topic.lower().split())
            # Maximize word-set difference
            overlap = len(current_words & topic_words)
            diff = len(topic_words) - overlap
            if diff > best_diff:
                best_diff = diff
                best_topic = topic

        return best_topic or self._random_seed_topic()

    def _next_seed_topic(self) -> str:
        self.current_topic_idx = (self.current_topic_idx + 1) % len(self.seed_topics)
        self._exploit_streak = 0
        return self.seed_topics[self.current_topic_idx]

    def _random_seed_topic(self) -> str:
        least_explored = None
        min_exp = float("inf")
        for topic in self.seed_topics:
            key = self._extract_topic(topic)
            exp = self.self_model.experience.get(key, 0)
            if exp < min_exp:
                min_exp = exp
                least_explored = topic
        return least_explored or self.seed_topics[0]

    def _extract_topic(self, query: str) -> str:
        words = [w for w in query.lower().split() if len(w) > 3][:3]
        return " ".join(words) if words else query[:30]

    def dream(self, memory=None) -> None:
        """Called after nightly dreaming — fine-tune PFC, reset fatigue, reflect."""

        # Merge duplicate topics before dreaming
        n_merged = self.self_model.merge_topics()
        if n_merged:
            log.info(f"Merged {n_merged} duplicate topics in self-model")

        # Fine-tune the prefrontal cortex
        try:
            from halo3.training.dream_finetune import dream_finetune
            findings = memory.get_findings() if memory else []
            success = dream_finetune(
                age=self.self_model.age,
                competence=self.self_model.competence,
                traits=self.self_model.traits,
                narrative=self.self_model.narrative,
                strengths=self.self_model.strengths,
                weaknesses=self.self_model.weaknesses,
                findings=findings,
                dead_queries=self.self_model.dead_queries,
            )
            if success:
                self.prefrontal.upgrade_to_organism_model()
                log.info("Prefrontal cortex now carries this organism's identity")
        except Exception as e:
            log.warning(f"Dream fine-tuning failed: {e}")

        self.drives.dream_reset()
        self.clock.mark_dreamed()

        # Deep self-reflection
        reflection = self.prefrontal.self_reflect(
            age=self.self_model.age,
            emotion_history=list(self.emotions.history),
            strengths=self.self_model.strengths,
            weaknesses=self.self_model.weaknesses,
            n_findings=sum(1 for n in self.self_model.narrative if "Discover" in n),
            narrative=self.self_model.narrative,
        )

        if reflection:
            self.self_model.narrative.append(
                f"[Tick {self.self_model.age}] Dream narrative (LLM): {reflection}"
            )
            log.info(f"Dream narrative (LLM-generated): {reflection}")
        else:
            self.self_model.narrative.append(
                f"[Tick {self.self_model.age}] Dreamed. "
                f"Dominant emotion: {self.emotions.dominant_recent}. "
                f"Identity: {self.self_model.identity_statement}"
            )

        # Generate post-dream exploration plan
        plan = self.prefrontal.generate_exploration_plan(
            self.seed_topics, self.self_model.strengths,
        )
        if plan:
            self._exploration_plan = plan
            log.info(f"Post-dream exploration plan: {plan}")

        self.self_model.save()
        log.info(f"Awoke. {self.self_model.identity_statement}")

    def status(self) -> str:
        base = (
            f"Age: {self.self_model.age} ticks | "
            f"{self.emotions.emoji()} {self.emotions.current} | "
            f"{self.self_model.identity_statement}"
        )
        if self.self_model.age > 0 and self.self_model.age % 10 == 0:
            reflection = self.prefrontal.self_reflect(
                age=self.self_model.age,
                emotion_history=list(self.emotions.history),
                strengths=self.self_model.strengths,
                weaknesses=self.self_model.weaknesses,
                n_findings=sum(1 for n in self.self_model.narrative if "Discover" in n),
                narrative=self.self_model.narrative,
            )
            if reflection:
                self.self_model.narrative.append(
                    f"[Tick {self.self_model.age}] Reflection: {reflection}"
                )
                log.info(f"  Self-narrative: {reflection}")
        return base
