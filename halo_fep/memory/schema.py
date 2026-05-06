"""Episode dataclass — one subconscious tick's experience."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Episode:
    query:              str
    tokens:             np.ndarray   # (n_tokens, d_model) float32
    swarm_mu:           np.ndarray   # (n_agents, n_hidden) float32
    free_energy:        float
    free_energy_delta:  float = 0.0
    llm_output:         str | None = None
    topic_tags:         list[str] = field(default_factory=list)
    id:                 str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:          float = field(default_factory=time.time)
