"""Active Sampler — Free Energy-guided curriculum selection.

Uses Black-Scholes valuation to pick topics, then forward-only prediction
error to filter candidates into the zone of proximal development.

Zone of proximal development: texts with medium FE are most informative.
  - Low FE = already mastered (boring)
  - High FE = incomprehensible noise (overwhelming)
  - Medium FE = complex enough to learn from, simple enough to integrate
"""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)


def select_texts_by_fe(
    texts: list[str],
    fe_scores: list[float],
    n_select: int = 10,
) -> list[str]:
    """Filter texts to the zone of proximal development based on FE scores.

    Removes bottom 20% (mastered) and top 20% (noise), then picks
    the top n_select from the remaining zone (highest FE within zone —
    prefer the most challenging learnable content).
    """
    if len(texts) <= n_select:
        return list(texts)

    paired = sorted(zip(texts, fe_scores), key=lambda x: x[1])
    lo = len(paired) // 5
    hi = 4 * len(paired) // 5
    if hi <= lo:
        hi = len(paired)
        lo = 0
    zone = paired[lo:hi]

    # Within the zone, prefer highest FE (most challenging but learnable)
    zone.sort(key=lambda x: -x[1])
    return [text for text, _ in zone[:n_select]]


def rank_topics_by_bs(
    topic_index,
    volatility_surface,
    n_top: int = 20,
) -> list[tuple[int, float]]:
    """Rank topic buckets by Black-Scholes option value.

    Returns list of (topic_id, bs_value) sorted descending.
    """
    topics = topic_index.get_topics()
    valued = []
    for topic in topics:
        primary_kw = topic.keywords[0] if topic.keywords else ""
        value = volatility_surface.value_topic(primary_kw)
        valued.append((topic.topic_id, value))
    valued.sort(key=lambda x: -x[1])
    return valued[:n_top]


def sample_candidates(
    topic_index,
    ranked_topics: list[tuple[int, float]],
    n_candidates: int = 50,
) -> list[str]:
    """Stream candidate texts from top-ranked topics."""
    if not ranked_topics:
        return []
    per_topic = max(1, n_candidates // len(ranked_topics))
    texts = []
    for topic_id, _ in ranked_topics:
        batch = topic_index.sample_from_topic(topic_id, n=per_topic)
        texts.extend(batch)
        if len(texts) >= n_candidates:
            break
    return texts[:n_candidates]


def select_curriculum(
    model,
    carry,
    topic_index,
    volatility_surface,
    embedder,
    n_candidates: int = 50,
    n_train: int = 10,
    key=None,
) -> list[str]:
    """Select the most informative texts for training.

    Full pipeline: BS ranks topics -> stream candidates -> FE scores -> zone filter.
    """
    import jax
    from halo3.loss import halo3_loss

    # Step 1: Rank topics by BS value
    ranked = rank_topics_by_bs(topic_index, volatility_surface, n_top=20)
    if not ranked:
        log.warning("ActiveSampler: no topics ranked — returning empty")
        return []

    # Step 2: Stream candidates from top topics
    candidates = sample_candidates(topic_index, ranked, n_candidates)
    if not candidates:
        log.warning("ActiveSampler: no candidates streamed — returning empty")
        return []

    log.info(f"ActiveSampler: scoring {len(candidates)} candidates by FE...")

    # Step 3: Forward-only FE scoring
    if key is None:
        key = jax.random.PRNGKey(42)

    fe_scores = []
    for text in candidates:
        tokens = embedder.texts_to_tokens([text], n_tokens=model.cfg.n_tokens)
        loss, _ = halo3_loss(model, carry, tokens, key)
        fe_scores.append(float(loss))

    # Step 4: Zone of proximal development
    selected = select_texts_by_fe(candidates, fe_scores, n_select=n_train)
    log.info(f"ActiveSampler: selected {len(selected)} texts (FE range: "
             f"{min(fe_scores):.2f} - {max(fe_scores):.2f})")
    return selected
