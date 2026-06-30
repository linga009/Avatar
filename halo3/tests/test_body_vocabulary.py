"""Tests for BodyVocabulary — body-language bridge."""
import pytest
import jax
import jax.numpy as jnp


class TestBodyVocabularyDecode:
    """Test the core _decode method."""

    def test_decode_random_vector_returns_words(self):
        """Random d_model vector should decode to valid word strings."""
        from halo3.body_vocabulary import BodyVocabulary
        bv = BodyVocabulary(
            lm_head_path="data/checkpoints/halo3_lm_lm.eqx",
            tokenizer_path="data/tokenizer.model",
        )
        if not bv.enabled:
            pytest.skip("LM head checkpoint not available")
        h = jax.random.normal(jax.random.PRNGKey(0), (4, 2048))
        words = bv._decode(h)
        assert isinstance(words, list)
        assert len(words) > 0
        assert all(isinstance(w, str) for w in words)

    def test_decode_zero_vector_returns_list(self):
        """Zero vector should still return a list (possibly empty)."""
        from halo3.body_vocabulary import BodyVocabulary
        bv = BodyVocabulary(
            lm_head_path="data/checkpoints/halo3_lm_lm.eqx",
            tokenizer_path="data/tokenizer.model",
        )
        if not bv.enabled:
            pytest.skip("LM head checkpoint not available")
        h = jnp.zeros((4, 2048))
        words = bv._decode(h)
        assert isinstance(words, list)

    def test_filter_special_tokens(self):
        """Decoded words should not contain pad/unk/bos/eos or short fragments."""
        from halo3.body_vocabulary import BodyVocabulary
        bv = BodyVocabulary(
            lm_head_path="data/checkpoints/halo3_lm_lm.eqx",
            tokenizer_path="data/tokenizer.model",
        )
        if not bv.enabled:
            pytest.skip("LM head checkpoint not available")
        h = jax.random.normal(jax.random.PRNGKey(42), (8, 2048))
        words = bv._decode(h)
        for w in words:
            assert len(w) >= 3, f"Short fragment not filtered: '{w}'"

    def test_top_k_limit(self):
        """Should return at most 15 words."""
        from halo3.body_vocabulary import BodyVocabulary
        bv = BodyVocabulary(
            lm_head_path="data/checkpoints/halo3_lm_lm.eqx",
            tokenizer_path="data/tokenizer.model",
        )
        if not bv.enabled:
            pytest.skip("LM head checkpoint not available")
        h = jax.random.normal(jax.random.PRNGKey(7), (16, 2048))
        words = bv._decode(h)
        assert len(words) <= 15

    def test_graceful_missing_checkpoint(self):
        """Missing checkpoint should disable bridge, not crash."""
        from halo3.body_vocabulary import BodyVocabulary
        bv = BodyVocabulary(
            lm_head_path="/nonexistent/path.eqx",
            tokenizer_path="/nonexistent/tokenizer.model",
        )
        assert not bv.enabled
        words = bv._decode(jnp.zeros((4, 2048)))
        assert words == []
        body_words = bv.get_body_words()
        assert body_words["thoughts"] == []


class TestTieredSchedule:
    """Test the tiered decode scheduling."""

    def _make_bv(self):
        from halo3.body_vocabulary import BodyVocabulary
        bv = BodyVocabulary(
            lm_head_path="data/checkpoints/halo3_lm_lm.eqx",
            tokenizer_path="data/tokenizer.model",
        )
        if not bv.enabled:
            pytest.skip("LM head checkpoint not available")
        return bv

    def test_thoughts_decode_every_tick(self):
        """h_out thoughts should update every tick."""
        bv = self._make_bv()
        h1 = jax.random.normal(jax.random.PRNGKey(1), (32, 2048))
        h2 = jax.random.normal(jax.random.PRNGKey(2), (32, 2048))
        cache = jnp.zeros((128, 2048))
        island = jnp.zeros((32, 2048))

        bv.tick(h1, cache, 0, island, surprise=0.0, tick_num=1)
        words1 = bv.get_body_words()["thoughts"]

        bv.tick(h2, cache, 0, island, surprise=0.0, tick_num=2)
        words2 = bv.get_body_words()["thoughts"]

        # Different inputs should produce different words
        assert words1 != words2

    def test_recent_only_every_5_ticks(self):
        """Recent thoughts should only update on tick 5, 10, 15..."""
        bv = self._make_bv()
        h = jax.random.normal(jax.random.PRNGKey(3), (32, 2048))
        cache = jax.random.normal(jax.random.PRNGKey(4), (128, 2048))
        island = jnp.zeros((32, 2048))

        # Tick 1: recent should remain empty
        bv.tick(h, cache, 64, island, surprise=0.0, tick_num=1)
        assert bv.get_body_words()["recent"] == []

        # Tick 5: recent should populate
        bv.tick(h, cache, 64, island, surprise=0.0, tick_num=5)
        assert len(bv.get_body_words()["recent"]) > 0

    def test_knowledge_only_every_50_ticks(self):
        """Deep knowledge should only update on tick 50, 100..."""
        bv = self._make_bv()
        h = jax.random.normal(jax.random.PRNGKey(5), (32, 2048))
        cache = jnp.zeros((128, 2048))
        island = jax.random.normal(jax.random.PRNGKey(6), (32, 2048))

        # Tick 1: knowledge should remain empty
        bv.tick(h, cache, 0, island, surprise=0.0, tick_num=1)
        assert bv.get_body_words()["knowledge"] == []

        # Tick 50: knowledge should populate
        bv.tick(h, cache, 0, island, surprise=0.0, tick_num=50)
        assert len(bv.get_body_words()["knowledge"]) > 0

    def test_surprise_below_threshold(self):
        """No surprise decode when self_surprise < 0.3."""
        bv = self._make_bv()
        h = jax.random.normal(jax.random.PRNGKey(7), (32, 2048))
        cache = jax.random.normal(jax.random.PRNGKey(8), (128, 2048))
        island = jnp.zeros((32, 2048))

        bv.tick(h, cache, 64, island, surprise=0.1, tick_num=1)
        assert bv.get_body_words()["expected"] == []
        assert bv.get_body_words()["actual"] == []

    def test_surprise_above_threshold(self):
        """Surprise decode should fire when self_surprise > 0.3."""
        bv = self._make_bv()
        h = jax.random.normal(jax.random.PRNGKey(9), (32, 2048))
        cache = jax.random.normal(jax.random.PRNGKey(10), (128, 2048))
        island = jnp.zeros((32, 2048))

        bv.tick(h, cache, 64, island, surprise=0.5, tick_num=1)
        assert len(bv.get_body_words()["expected"]) > 0
        assert len(bv.get_body_words()["actual"]) > 0
