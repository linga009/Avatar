"""Self-Model — the organism's representation of itself.

Tracks what topics the organism is good at (high r historically),
what it struggles with (low r), and builds a narrative identity
from accumulated experience.

v3.1 fixes:
  - Weakness threshold raised from 0.3 to 0.48 (reachable with EMA alpha=0.95)
  - Competence decay: unvisited topics drift toward 0.5 over time
  - Topic merging: canonicalizes duplicate topic keys
  - Meta-cognitive tracking: query success rate
  - Dead query tracking: remembers queries that returned nothing
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import os


@dataclass
class SelfModel:
    """The organism's evolving self-representation."""

    competence: dict[str, float] = field(default_factory=dict)
    experience: dict[str, int] = field(default_factory=dict)
    narrative: list[str] = field(default_factory=list)
    traits: dict[str, float] = field(default_factory=dict)
    age: int = 0
    # Meta-cognition
    _query_results: dict[str, int] = field(default_factory=dict)  # query -> 0=fail, 1=success count
    dead_queries: list[str] = field(default_factory=list)  # queries that consistently return nothing
    _last_topic: str = ""

    def update(self, topic: str, r_mean: float, emotion: str, finding: str | None) -> None:
        self.age += 1

        # Competence EMA
        alpha = 0.95
        prev = self.competence.get(topic, 0.5)
        self.competence[topic] = alpha * prev + (1.0 - alpha) * r_mean
        self.experience[topic] = self.experience.get(topic, 0) + 1

        # Competence decay: topics not visited this tick drift toward 0.5
        decay_rate = 0.999
        for t in list(self.competence.keys()):
            if t != topic:
                c = self.competence[t]
                self.competence[t] = c * decay_rate + 0.5 * (1.0 - decay_rate)

        # Trait updates
        self.traits["curiosity_tendency"] = self.traits.get("curiosity_tendency", 0.5) * 0.99 + \
            (0.01 if emotion == "curiosity" else 0.0)
        self.traits["anxiety_tendency"] = self.traits.get("anxiety_tendency", 0.0) * 0.99 + \
            (0.01 if emotion == "anxiety" else 0.0)
        self.traits["persistence"] = self.traits.get("persistence", 0.5) * 0.99 + \
            (0.01 if emotion == "satisfaction" else 0.0)
        self.traits["frustration_tendency"] = self.traits.get("frustration_tendency", 0.0) * 0.99 + \
            (0.01 if emotion == "frustration" else 0.0)

        self._last_topic = topic

        if finding:
            self.narrative.append(
                f"[Tick {self.age}] Discovered: {finding[:80]}... "
                f"(felt {emotion}, r={r_mean:.3f})"
            )
            if len(self.narrative) > 200:
                self.narrative = self.narrative[-200:]

    def record_dead_query(self, query: str) -> None:
        """Record a query that consistently returned zero results."""
        if query not in self.dead_queries:
            self.dead_queries.append(query)
            # Keep bounded
            if len(self.dead_queries) > 50:
                self.dead_queries = self.dead_queries[-50:]

    def merge_topics(self) -> int:
        """Merge duplicate topic keys by word-set similarity.

        Returns number of merges performed.
        """
        keys = list(self.competence.keys())
        merged = 0
        seen = {}  # canonical_key -> original_key

        for k in keys:
            canonical = " ".join(sorted(set(k.lower().split())))
            if canonical in seen:
                # Merge into existing
                existing = seen[canonical]
                # Weighted average by experience
                e1 = self.experience.get(existing, 1)
                e2 = self.experience.get(k, 1)
                total = e1 + e2
                self.competence[existing] = (
                    self.competence.get(existing, 0.5) * e1 +
                    self.competence.get(k, 0.5) * e2
                ) / total
                self.experience[existing] = total
                # Remove duplicate
                del self.competence[k]
                if k in self.experience:
                    del self.experience[k]
                merged += 1
            else:
                seen[canonical] = k

        return merged

    @property
    def strengths(self) -> list[str]:
        """Topics where competence > 0.5, sorted by competence."""
        return sorted(
            [t for t, c in self.competence.items() if c > 0.5],
            key=lambda t: self.competence[t],
            reverse=True,
        )

    @property
    def weaknesses(self) -> list[str]:
        """Topics where competence < 0.48.

        FIXED: threshold raised from 0.3 to 0.48 so this is actually
        reachable with EMA alpha=0.95 starting from 0.5 baseline.
        """
        return sorted(
            [t for t, c in self.competence.items() if c < 0.48],
            key=lambda t: self.competence[t],
        )

    @property
    def identity_statement(self) -> str:
        if self.age < 10:
            return "I am newly awakened, still learning what I am."

        strong = self.strengths[:3]
        n_findings = sum(1 for n in self.narrative if "Discovered" in n)

        if strong:
            topics = ", ".join(strong)
            return (
                f"I am a {self.age}-tick-old research mind. "
                f"I resonate most with {topics}. "
                f"I have made {n_findings} discoveries."
            )
        return f"I am a {self.age}-tick-old research mind, still finding my resonance."

    def save(self, path: str = "data/self_model.json") -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "age": self.age,
            "competence": self.competence,
            "experience": self.experience,
            "traits": self.traits,
            "narrative": self.narrative[-50:],
            "dead_queries": self.dead_queries[-50:],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str = "data/self_model.json") -> SelfModel:
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        sm = cls()
        sm.age = data.get("age", 0)
        sm.competence = data.get("competence", {})
        sm.experience = data.get("experience", {})
        sm.traits = data.get("traits", {})
        sm.narrative = data.get("narrative", [])
        sm.dead_queries = data.get("dead_queries", [])
        return sm
