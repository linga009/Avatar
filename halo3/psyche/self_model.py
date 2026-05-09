"""Self-Model — the organism's representation of itself.

Tracks what topics the organism is good at (high r historically),
what it struggles with (low r), and builds a narrative identity
from accumulated experience.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os


@dataclass
class SelfModel:
    """The organism's evolving self-representation."""

    # Topic competence: topic → running average of r
    competence: dict[str, float] = field(default_factory=dict)
    # Topic experience count
    experience: dict[str, int] = field(default_factory=dict)
    # Narrative fragments: key moments
    narrative: list[str] = field(default_factory=list)
    # Core identity traits (emerge from experience)
    traits: dict[str, float] = field(default_factory=dict)
    # Total ticks lived
    age: int = 0

    def update(self, topic: str, r_mean: float, emotion: str, finding: str | None) -> None:
        """Update self-model after a tick."""
        self.age += 1

        # Update competence (EMA)
        alpha = 0.95
        prev = self.competence.get(topic, 0.5)
        self.competence[topic] = alpha * prev + (1.0 - alpha) * r_mean
        self.experience[topic] = self.experience.get(topic, 0) + 1

        # Update traits based on emotional patterns
        self.traits["curiosity_tendency"] = self.traits.get("curiosity_tendency", 0.5) * 0.99 + \
            (0.01 if emotion == "curiosity" else 0.0)
        self.traits["anxiety_tendency"] = self.traits.get("anxiety_tendency", 0.0) * 0.99 + \
            (0.01 if emotion == "anxiety" else 0.0)
        self.traits["persistence"] = self.traits.get("persistence", 0.5) * 0.99 + \
            (0.01 if emotion == "satisfaction" else 0.0)

        # Add narrative moments for significant events
        if finding:
            self.narrative.append(
                f"[Tick {self.age}] Discovered: {finding[:80]}... "
                f"(felt {emotion}, r={r_mean:.3f})"
            )
            # Keep narrative bounded
            if len(self.narrative) > 200:
                self.narrative = self.narrative[-200:]

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
        """Topics where competence < 0.3."""
        return [t for t, c in self.competence.items() if c < 0.3]

    @property
    def identity_statement(self) -> str:
        """One-sentence self-description based on accumulated experience."""
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
        """Persist self-model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "age": self.age,
            "competence": self.competence,
            "experience": self.experience,
            "traits": self.traits,
            "narrative": self.narrative[-50:],  # last 50 entries
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str = "data/self_model.json") -> SelfModel:
        """Load self-model from disk."""
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
        return sm
