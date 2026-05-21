"""Test Episode schema with codebook indices."""
from halo3.memory.schema import Episode


def test_episode_has_code_fields():
    ep = Episode(
        query="test", order_param=0.5, mode="curiosity",
        audio_codes=[0, 1, 2, 3, 4, 5, 6, 7],
        vision_codes=[0, 1, 2, 3],
    )
    assert ep.audio_codes == [0, 1, 2, 3, 4, 5, 6, 7]
    assert ep.vision_codes == [0, 1, 2, 3]


def test_episode_codes_default_none():
    ep = Episode(query="test", order_param=0.5, mode="curiosity")
    assert ep.audio_codes is None
    assert ep.vision_codes is None
