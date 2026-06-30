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
