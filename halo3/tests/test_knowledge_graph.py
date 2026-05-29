# halo3/tests/test_knowledge_graph.py
from halo3.psyche.knowledge_graph import KnowledgeGraph


def test_add_discovery_creates_node():
    g = KnowledgeGraph()
    g.add_discovery(topic_key="quantum error", finding="correlation between error rates and topology",
                    r_mean=0.65, chi=0.4, emotion="pride")
    assert g.node_count == 1
    assert "quantum error" in g.topics


def test_add_two_discoveries_creates_edge_if_related():
    g = KnowledgeGraph()
    g.add_discovery(topic_key="quantum error", finding="error correction codes",
                    r_mean=0.65, chi=0.4, emotion="pride")
    g.add_discovery(topic_key="quantum computing", finding="computing with error correction",
                    r_mean=0.62, chi=0.5, emotion="curiosity")
    assert g.edge_count >= 1
    assert g.has_edge("quantum error", "quantum computing")


def test_no_edge_for_unrelated_topics():
    g = KnowledgeGraph()
    g.add_discovery(topic_key="quantum error", finding="error correction",
                    r_mean=0.65, chi=0.4, emotion="pride",
                    timestamp="2026-01-01T00:00:00")
    g.add_discovery(topic_key="climate modeling", finding="atmospheric patterns",
                    r_mean=0.61, chi=0.3, emotion="satisfaction",
                    timestamp="2026-05-01T00:00:00")
    assert not g.has_edge("quantum error", "climate modeling")


def test_topology_metrics():
    g = KnowledgeGraph()
    g.add_discovery(topic_key="quantum error", finding="a", r_mean=0.65, chi=0.4, emotion="pride")
    g.add_discovery(topic_key="quantum computing", finding="b", r_mean=0.62, chi=0.5, emotion="curiosity")
    g.add_discovery(topic_key="quantum networks", finding="c", r_mean=0.63, chi=0.3, emotion="satisfaction")
    metrics = g.get_topology_metrics()
    assert "density" in metrics
    assert "frontier_size" in metrics
    assert "avg_clustering" in metrics
    assert 0.0 <= metrics["density"] <= 1.0


def test_node_metrics():
    g = KnowledgeGraph()
    g.add_discovery(topic_key="quantum error", finding="a", r_mean=0.65, chi=0.4, emotion="pride")
    g.add_discovery(topic_key="quantum computing", finding="b", r_mean=0.62, chi=0.5, emotion="curiosity")
    nm = g.get_node_metrics("quantum error")
    assert "degree" in nm
    assert "clustering_coeff" in nm


def test_node_metrics_unknown_topic():
    g = KnowledgeGraph()
    nm = g.get_node_metrics("nonexistent")
    assert nm["degree"] == 0
    assert nm["is_frontier"] is True


def test_save_and_load(tmp_path):
    g = KnowledgeGraph()
    g.add_discovery(topic_key="quantum error", finding="a", r_mean=0.65, chi=0.4, emotion="pride")
    g.add_discovery(topic_key="quantum computing", finding="b", r_mean=0.62, chi=0.5, emotion="curiosity")
    path = str(tmp_path / "graph.json")
    g.save(path)
    g2 = KnowledgeGraph.load(path)
    assert g2.node_count == 2
    assert g2.edge_count == g.edge_count


def test_prune_weak_edges():
    g = KnowledgeGraph()
    g.add_discovery(topic_key="quantum error", finding="a", r_mean=0.65, chi=0.4, emotion="pride")
    g.add_discovery(topic_key="quantum computing", finding="b", r_mean=0.62, chi=0.5, emotion="curiosity")
    for u, v in g._graph.edges():
        g._graph[u][v]["weight"] = 0.05
    pruned = g.prune_weak_edges(threshold=0.1)
    assert pruned >= 1
    assert g.edge_count == 0


def test_strengthen_recent():
    g = KnowledgeGraph()
    g.add_discovery(topic_key="quantum error", finding="a", r_mean=0.65, chi=0.4, emotion="pride")
    g.add_discovery(topic_key="quantum computing", finding="b", r_mean=0.62, chi=0.5, emotion="curiosity")
    if g.edge_count > 0:
        old_weight = list(g._graph.edges(data=True))[0][2]["weight"]
        g.strengthen_recent("quantum error", factor=1.5)
        new_weight = list(g._graph.edges(data=True))[0][2]["weight"]
        assert new_weight >= old_weight
