# Knowledge Graph for Avatar — Design Rationale

**Date:** 2026-05-29
**Author:** Dr. Linga Murthy Narlagiri

---

## Why Avatar Needs a Knowledge Graph

Avatar has processed 4,812 episodes across 155 topic domains and made 840 discoveries (r > 0.6). But it has no memory of how those discoveries relate to each other. Each finding is stored as a flat record in SQLite — a row with a topic, a score, and a timestamp. There are no connections between discoveries.

This means Avatar cannot answer: "What connects topic X to topic Y?" It cannot recognise when a new discovery bridges two previously unrelated domains. It cannot direct its exploration toward knowledge gaps because it has no map of what it knows.

Human memory is associative and graph-structured. Concepts link to related concepts. Clusters form around areas of expertise. Bridges between clusters represent interdisciplinary insight. Orphan nodes represent unexplored territory. Avatar's flat episode store is more like a filing cabinet than a mind.

The knowledge graph gives Avatar a topological map of its own understanding.

---

## What the Knowledge Graph Is

A lightweight, persistent graph (NetworkX + JSON) where:

- **Nodes** are topics that Avatar has explored. Each node stores: topic key, discovery count, average order parameter r, best r achieved, associated affect states, timestamps.

- **Edges** connect related topics. Edge weight combines three signals:
  - Semantic overlap (40%): shared words between topic keys
  - Temporal proximity (30%): topics explored within 3 days of each other
  - Finding mentions (30%): one discovery's text references another topic

- **Topology metrics** are computed every 10 ticks and describe the shape of Avatar's knowledge: density, clustering coefficient, frontier size, community structure, bridge nodes.

The graph persists to `data/checkpoints/knowledge_graph.json` and survives restarts.

---

## What It Achieves

### 1. Directed Exploration Instead of Random Topic Rotation

Today, when Avatar is bored, it picks the next topic from Black-Scholes valuation. This treats topics independently — each option is priced without knowing how it relates to others.

With the knowledge graph, the volatility surface gains a new signal: **frontier proximity**. Topics connected to unexplored regions (low-degree nodes, graph frontier) get an implied volatility boost. Topics in dense, well-understood clusters get a penalty.

Result: Avatar explores toward the edges of its knowledge, not randomly.

### 2. Cross-Domain Surprise Detection

When Avatar discovers something new (r > 0.6), the graph checks whether this discovery creates a short path between previously distant nodes. If topic A and topic B were 5 hops apart and a new discovery connects them in 2 hops, that is a structural surprise — an interdisciplinary bridge.

This graph-based surprise feeds into the introspective monitor. Avatar can notice: "I just connected X and Y in a way I hadn't before." This is a signal the current flat episode store cannot produce.

### 3. Satiation That Reflects Understanding Depth

Currently, satiation triggers when r > 0.55 and chi < 0.2 — meaning the oscillators are synchronised and rigid. But this is a physics signal, not a knowledge signal. Avatar might be physically synchronised on a topic it has barely explored.

With graph metrics, satiation also considers local clustering coefficient. If the topic's neighbourhood is dense (clustering > 0.8), the topic is genuinely well-mapped and satiation accelerates. If the neighbourhood is sparse, there is more to discover even if the physics body is temporarily synchronised.

### 4. Dream Consolidation That Reorganises Knowledge

During the dream cycle, a new consolidation phase traverses the graph:
- Prune weak edges (weight < 0.1) — forget tenuous connections
- Strengthen edges to recent discoveries — reinforce what was just learned
- Detect communities — identify topic clusters that Avatar has developed expertise in
- Log the graph state — dream narrative includes "I consolidated 3 knowledge clusters"

This mirrors biological sleep consolidation, where the hippocampus replays experiences and strengthens or prunes synaptic connections.

### 5. Proactive Notifications That Are Actually Interesting

Currently, Avatar sends proactive messages like "I discovered something about X." With the knowledge graph, it can send: "I just found a connection between X and Y that I hadn't seen before — together they suggest Z."

This is the difference between a filing clerk and a research partner.

---

## How It Integrates Without Changing the Physics

The knowledge graph does NOT replace COP, drives, or the volatility surface. It sits alongside them:

| Component | What it measures | Graph's role |
|---|---|---|
| COP (chi, tau, unity) | Physics dynamics of the oscillator system | Unchanged — graph metrics are orthogonal |
| Drives (curiosity, satiation) | Behavioural needs | Graph modulates: frontier proximity boosts curiosity, clustering accelerates satiation |
| Volatility (Black-Scholes) | Topic option value | Graph adjusts implied volatility: frontier topics get IV boost |
| Introspection | Self-surprise | Graph adds structural surprise: unexpected bridges between distant nodes |

COP tells Avatar what its body IS doing. The graph tells Avatar what it COULD explore. These are complementary, not competing.

---

## Resource Budget

| Resource | Cost | Budget | Safe? |
|---|---|---|---|
| RAM | 15-25 MB (1000 nodes, 5000 edges) | 300 MB headroom | Yes |
| GPU VRAM | 0 MB (CPU only) | — | Yes |
| CPU per tick | < 1ms (metric lookup from cache) | — | Yes |
| CPU per 10 ticks | ~10ms (topology recomputation) | — | Yes |
| Disk | ~200 KB (JSON serialisation) | — | Yes |

The knowledge graph adds negligible overhead. NetworkX is a pure Python library already available in the Docker image.

---

## What Success Looks Like

After implementing the knowledge graph, we expect:

1. **Topic diversity increases** — frontier-guided exploration visits more domains than random BS valuation
2. **Discovery rate improves** — exploring near frontiers means working in the zone of proximal development
3. **Cross-domain discoveries emerge** — graph bridges detected between previously unconnected topics
4. **Satiation is more accurate** — physics synchronisation alone no longer triggers premature topic switching
5. **Dream consolidation produces measurable graph changes** — communities form, weak edges pruned, knowledge structure visible

These are testable in the ablation framework. We can run 200 ticks with and without the knowledge graph and compare topic diversity, discovery rate, and graph topology metrics.

---

## Philosophical Grounding

The knowledge graph is the computational analogue of what Bohm called "soma-significance" — the body's understanding feeding back into meaningful structure. Avatar's physics body produces order parameters, free energy, and synchronisation. The knowledge graph transforms those measurements into a map of what the system has learned and where it should look next.

It is also the minimal implementation of Maturana and Varela's autopoietic requirement for self-knowledge: an autonomous system must maintain a model of its own competence boundaries. Without the graph, Avatar explores blindly. With it, Avatar explores with a map.

This is not Obsidian. It is not RAG. It is not a vector database. It is a 25 MB graph that gives Avatar the one thing it currently lacks: knowing what it knows.
