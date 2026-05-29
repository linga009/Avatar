"""Knowledge graph — topological map of Avatar's discoveries.

Nodes are topics where r > 0.6 (discoveries). Edges connect related
topics by semantic overlap, temporal proximity, and finding mentions.
Topology metrics feed into drives and volatility for directed exploration.

Resource cost: ~15-25 MB for 1000 nodes x 5000 edges. CPU only.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import networkx as nx

log = logging.getLogger(__name__)

_SEMANTIC_WEIGHT = 0.4
_TEMPORAL_WEIGHT = 0.3
_MENTION_WEIGHT = 0.3
_TEMPORAL_DECAY_DAYS = 3.0
_MIN_EDGE_WEIGHT = 0.15


class KnowledgeGraph:
    """Lightweight discovery graph with topology metrics."""

    def __init__(self) -> None:
        self._graph = nx.Graph()
        self._discovery_times: dict[str, str] = {}
        self._metrics_cache: dict | None = None
        self._cache_tick: int = -1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def topics(self) -> list[str]:
        return list(self._graph.nodes())

    def has_edge(self, a: str, b: str) -> bool:
        return self._graph.has_edge(a, b)

    # ------------------------------------------------------------------
    # Core: add discovery
    # ------------------------------------------------------------------

    def add_discovery(
        self,
        topic_key: str,
        finding: str,
        r_mean: float,
        chi: float,
        emotion: str,
        timestamp: str | None = None,
    ) -> None:
        """Add or update a discovery node, then link to related nodes."""
        ts = timestamp or datetime.now().isoformat()

        # --- upsert node ---
        if self._graph.has_node(topic_key):
            node = self._graph.nodes[topic_key]
            node["n_discoveries"] = node.get("n_discoveries", 0) + 1
            node["avg_r"] = node.get("avg_r", r_mean) * 0.7 + r_mean * 0.3
            node["max_r"] = max(node.get("max_r", 0), r_mean)
            node["last_visited"] = ts
            node["last_emotion"] = emotion
        else:
            self._graph.add_node(
                topic_key,
                n_discoveries=1,
                avg_r=r_mean,
                max_r=r_mean,
                first_discovery=ts,
                last_visited=ts,
                last_emotion=emotion,
                last_chi=chi,
            )
            self._discovery_times[topic_key] = ts

        # --- create edges to existing nodes ---
        topic_words = set(topic_key.lower().split())
        finding_lower = finding.lower() if finding else ""

        for other in list(self._graph.nodes()):
            if other == topic_key:
                continue
            other_words = set(other.lower().split())

            # 1. Semantic overlap (Jaccard on key words)
            overlap = len(topic_words & other_words)
            union = len(topic_words | other_words)
            semantic = overlap / union if union > 0 else 0.0

            # 2. Temporal proximity (decay over days)
            other_ts = self._discovery_times.get(other, "")
            temporal = 0.0
            if other_ts and ts:
                try:
                    delta_days = abs(
                        (datetime.fromisoformat(ts) - datetime.fromisoformat(other_ts)).total_seconds()
                    ) / 86400
                    temporal = max(0.0, 1.0 - delta_days / _TEMPORAL_DECAY_DAYS)
                except (ValueError, TypeError):
                    pass

            # 3. Mention overlap (does the finding mention words from other topic?)
            mention = (
                1.0
                if any(w in finding_lower for w in other_words if len(w) >= 4)
                else 0.0
            )

            weight = (
                _SEMANTIC_WEIGHT * semantic
                + _TEMPORAL_WEIGHT * temporal
                + _MENTION_WEIGHT * mention
            )

            if weight >= _MIN_EDGE_WEIGHT:
                if self._graph.has_edge(topic_key, other):
                    self._graph[topic_key][other]["weight"] = max(
                        self._graph[topic_key][other]["weight"], weight
                    )
                else:
                    self._graph.add_edge(topic_key, other, weight=weight)

        self._metrics_cache = None

    # ------------------------------------------------------------------
    # Topology metrics
    # ------------------------------------------------------------------

    def get_topology_metrics(self, tick: int = 0) -> dict:
        """Global graph metrics. Cached within 10 ticks."""
        if self._metrics_cache and abs(tick - self._cache_tick) < 10:
            return self._metrics_cache

        n = self.node_count
        if n == 0:
            return {
                "density": 0.0,
                "avg_clustering": 0.0,
                "frontier_size": 0,
                "frontier_ratio": 0.0,
                "n_communities": 0,
                "giant_component_ratio": 0.0,
                "n_nodes": 0,
                "n_edges": 0,
            }

        density = nx.density(self._graph)
        avg_clustering = nx.average_clustering(self._graph) if n > 1 else 0.0

        # Frontier = low-degree nodes (leaf or isolated), ripe for exploration
        frontier = [nd for nd, deg in self._graph.degree() if deg <= 1]

        components = list(nx.connected_components(self._graph))
        giant = max(len(c) for c in components) if components else 0

        self._metrics_cache = {
            "density": round(density, 4),
            "avg_clustering": round(avg_clustering, 4),
            "frontier_size": len(frontier),
            "frontier_ratio": round(len(frontier) / max(n, 1), 4),
            "n_communities": len(components),
            "giant_component_ratio": round(giant / n, 4),
            "n_nodes": n,
            "n_edges": self.edge_count,
        }
        self._cache_tick = tick
        return self._metrics_cache

    def get_node_metrics(self, topic_key: str) -> dict:
        """Per-node metrics for drive/volatility integration."""
        if not self._graph.has_node(topic_key):
            return {"degree": 0, "clustering_coeff": 0.0, "is_frontier": True}

        degree = self._graph.degree(topic_key)
        clustering = nx.clustering(self._graph, topic_key)
        return {
            "degree": degree,
            "clustering_coeff": round(clustering, 4),
            "is_frontier": degree <= 1,
            "avg_r": self._graph.nodes[topic_key].get("avg_r", 0.0),
            "n_discoveries": self._graph.nodes[topic_key].get("n_discoveries", 0),
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune_weak_edges(self, threshold: float = 0.1) -> int:
        """Remove edges below weight threshold. Returns count removed."""
        to_remove = [
            (u, v)
            for u, v, d in self._graph.edges(data=True)
            if d.get("weight", 0) < threshold
        ]
        self._graph.remove_edges_from(to_remove)
        self._metrics_cache = None
        return len(to_remove)

    def strengthen_recent(self, topic_key: str, factor: float = 1.2) -> None:
        """Boost edge weights around a recently-visited topic."""
        if not self._graph.has_node(topic_key):
            return
        for neighbor in self._graph.neighbors(topic_key):
            self._graph[topic_key][neighbor]["weight"] = min(
                1.0, self._graph[topic_key][neighbor]["weight"] * factor
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize graph to JSON."""
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        data = {
            "nodes": {n: dict(self._graph.nodes[n]) for n in self._graph.nodes()},
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in self._graph.edges(data=True)
            ],
            "discovery_times": self._discovery_times,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("KnowledgeGraph saved: %d nodes, %d edges -> %s", self.node_count, self.edge_count, path)

    @classmethod
    def load(cls, path: str) -> KnowledgeGraph:
        """Deserialize graph from JSON."""
        g = cls()
        if not os.path.exists(path):
            log.warning("KnowledgeGraph file not found: %s — starting empty", path)
            return g
        with open(path) as f:
            data = json.load(f)
        for topic, attrs in data.get("nodes", {}).items():
            g._graph.add_node(topic, **attrs)
        for edge in data.get("edges", []):
            e = dict(edge)
            src, tgt = e.pop("source"), e.pop("target")
            g._graph.add_edge(src, tgt, **e)
        g._discovery_times = data.get("discovery_times", {})
        log.info("KnowledgeGraph loaded: %d nodes, %d edges from %s", g.node_count, g.edge_count, path)
        return g
