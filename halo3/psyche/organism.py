"""Organism — the unified psyche that lives in the physics body.

Integrates drives, emotions, self-model, circadian rhythm, and
prefrontal cortex (Qwen3 1.7B via Ollama) into a single coherent
agent that modulates the physics engine.

The limbic system (Kuramoto + drives + emotions) provides felt experience.
The prefrontal cortex (LLM) provides cognitive interpretation and planning.
"""
from __future__ import annotations
import logging
from halo3.psyche.drives import DriveState
from halo3.psyche.emotions import EmotionState
from halo3.psyche.self_model import SelfModel
from halo3.psyche.circadian import CircadianClock
from halo3.psyche.prefrontal import PrefrontalCortex

log = logging.getLogger(__name__)


class Organism:
    """The living psyche that inhabits the HoloBiont physics body.

    Each tick:
      1. Physics engine produces (r_mean, fe_delta)
      2. Organism updates drives, emotions, self-model
      3. Organism decides what to do next (topic, intensity, mode)
      4. Organism modulates physics (coupling K, search behavior)
    """

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

    def tick(
        self,
        r_mean: float,
        fe_delta: float,
        texts: list[str],
        current_query: str,
    ) -> dict:
        """Process one tick of lived experience.

        Returns dict with: emotion, next_query, coupling_mod, finding, log_line
        """
        # 1. Update drives
        self.drives.update(r_mean, fe_delta)

        # 2. Compute emotion
        emotion, intensity = self.emotions.update(r_mean, fe_delta)

        # 3. Determine finding — use prefrontal cortex to interpret
        finding = None
        if r_mean > 0.6 and texts:
            pfc_finding = self.prefrontal.interpret_finding(texts, current_query, r_mean)
            finding = pfc_finding or f"{'; '.join(texts[:3])}"

        # 4. Update self-model
        topic_key = self._extract_topic(current_query)
        self.self_model.update(topic_key, r_mean, emotion, finding)

        # 5. Decide next action — prefrontal cortex assists if available
        pfc_query = self.prefrontal.generate_query(
            current_query, emotion, r_mean, texts, self.self_model.strengths
        )
        if pfc_query:
            next_query = pfc_query
            log.debug(f"Prefrontal: generated query '{pfc_query}'")
        else:
            next_query = self._decide_next_query(emotion, r_mean, current_query, texts)

        # 6. Modulate physics
        coupling_mod = self.clock.modulate_coupling(1.0, self.drives.fatigue)

        # 7. Build log line
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
        }

    def _decide_next_query(
        self, emotion: str, r_mean: float, current_query: str, texts: list[str]
    ) -> str:
        """Emotion-driven query selection — the organism's will."""

        if emotion == "satisfaction":
            # Content — stay on topic but slow down (handled by tick interval)
            self._exploit_streak += 1
            if self._exploit_streak > 5:
                # Satiated, naturally move on
                return self._next_seed_topic()
            return current_query

        elif emotion == "pride":
            # Novel discovery confirmed — dig deeper
            self._exploit_streak += 1
            if texts:
                refinement = " ".join(texts[0].split()[:4])
                return current_query + " " + refinement
            return current_query

        elif emotion == "curiosity":
            # Edge of understanding — keep exploring this area
            self._exploit_streak = 0
            return current_query

        elif emotion == "boredom":
            # Nothing happening — jump to something completely different
            self._exploit_streak = 0
            # Don't just rotate — pick based on least-explored or strongest interest
            if self.self_model.weaknesses:
                return self.self_model.weaknesses[0]
            return self._random_seed_topic()

        elif emotion == "anxiety":
            # Overwhelmed — retreat to familiar territory
            self._exploit_streak = 0
            if self.self_model.strengths:
                return self.self_model.strengths[0]
            return self._next_seed_topic()

        return self._next_seed_topic()

    def _next_seed_topic(self) -> str:
        self.current_topic_idx = (self.current_topic_idx + 1) % len(self.seed_topics)
        self._exploit_streak = 0
        return self.seed_topics[self.current_topic_idx]

    def _random_seed_topic(self) -> str:
        """Pick a topic the organism hasn't explored much."""
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
        """Extract a stable topic key from a query string."""
        # Use first 3 significant words
        words = [w for w in query.lower().split() if len(w) > 3][:3]
        return " ".join(words) if words else query[:30]

    def dream(self, memory=None) -> None:
        """Called after nightly dreaming — fine-tune PFC, reset fatigue, reflect.

        This is where the LLM becomes part of the organism: its identity,
        memories, and competence are baked into a custom model.
        """
        # --- Fine-tune the prefrontal cortex on organism's experience ---
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
            )
            if success:
                self.prefrontal.upgrade_to_organism_model()
                log.info("Prefrontal cortex now carries this organism's identity")
        except Exception as e:
            log.warning(f"Dream fine-tuning failed: {e}")

        self.drives.dream_reset()
        self.clock.mark_dreamed()

        # Deep self-reflection via (now personalized) prefrontal cortex
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

        self.self_model.save()
        log.info(f"Awoke. {self.self_model.identity_statement}")

    def status(self) -> str:
        """Full organism status for display."""
        base = (
            f"Age: {self.self_model.age} ticks | "
            f"{self.emotions.emoji()} {self.emotions.current} | "
            f"{self.self_model.identity_statement}"
        )
        # Every 10 ticks, do a brief self-reflection
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
                log.info(f"  📋 Self-narrative (LLM-generated, not sentient): {reflection}")
        return base
