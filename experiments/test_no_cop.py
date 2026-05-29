from experiments.no_cop import NullCOP
from halo3.config import Halo3Config


def test_null_cop_returns_fixed_values():
    cop = NullCOP(Halo3Config())
    result = cop.observe(r_mean=0.3, r_a=0.4, r_c=0.2,
                         fe_delta=-0.1, K=0.3, theta=None)
    assert result["chi"] == 0.5
    assert result["tau"] == 0.5
    assert result["K_new"] == 0.3
    assert result["unity"] == 0.5


def test_null_cop_preserves_K():
    cop = NullCOP(Halo3Config())
    result = cop.observe(r_mean=0.1, r_a=0.1, r_c=0.1,
                         fe_delta=0.5, K=0.8, theta=None)
    assert result["K_new"] == 0.8
