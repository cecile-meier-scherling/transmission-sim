import numpy as np
import pytest
from jax import numpy as jnp

from examples.malaria_analysis import (
    exact_lineage_expectation,
    initial_state_m_equals_one,
    lineage_expectation_formula,
    mean_allele_frequency,
    prevalence,
)


def test_metrics_on_handbuilt_histogram():
    counts = jnp.asarray(
        [
            [6, 2, 1, 1],
            [0, 5, 5, 0],
        ],
        dtype=jnp.int32,
    )
    state_num_mutants = jnp.asarray([0, 1, 2, 1], dtype=jnp.int32)
    state_num_alleles = jnp.asarray([2, 2, 2, 4], dtype=jnp.int32)

    observed_mean = np.asarray(
        mean_allele_frequency(counts, state_num_mutants, state_num_alleles)
    )
    observed_prev = np.asarray(prevalence(counts, state_num_mutants))

    expected_mean = np.asarray([0.225, 0.75], dtype=np.float32)
    expected_prev = np.asarray([0.4, 1.0], dtype=np.float32)
    np.testing.assert_allclose(observed_mean, expected_mean, atol=1e-7)
    np.testing.assert_allclose(observed_prev, expected_prev, atol=1e-7)


def test_lineage_formula_spot_check():
    observed = float(lineage_expectation_formula(Q=4, sw=-0.2, st=0.5, chi=0.3))
    expected = (4.0 * (1.0 - 0.2) / (4.0 - 0.2)) * (1.0 + 0.5 / 4.0) * (1.0 + 0.3)
    np.testing.assert_allclose(observed, expected, atol=1e-7)


@pytest.mark.parametrize("Q", [4, 8, 16])
@pytest.mark.parametrize("sw", [-0.2, -0.1, 0.0])
def test_exact_lineage_expectation_matches_formula_for_sw_sweep(Q: int, sw: float):
    state = initial_state_m_equals_one(
        chi=0.0,
        sw=sw,
        st=0.0,
        num_hosts=4096,
        Q=Q,
    )
    observed = float(exact_lineage_expectation(state)[0])
    expected = float(lineage_expectation_formula(Q=Q, sw=sw, st=0.0, chi=0.0))
    np.testing.assert_allclose(observed, expected, rtol=0.05, atol=1e-5)


@pytest.mark.parametrize("Q", [4, 8, 16])
@pytest.mark.parametrize("st", [0.0, 0.25, 0.5, 1.0])
def test_exact_lineage_expectation_matches_st_scaling(Q: int, st: float):
    state = initial_state_m_equals_one(
        chi=0.0,
        sw=0.0,
        st=st,
        num_hosts=4096,
        Q=Q,
    )
    observed = float(exact_lineage_expectation(state)[0])
    expected = float(lineage_expectation_formula(Q=Q, sw=0.0, st=st, chi=0.0))
    np.testing.assert_allclose(observed, expected, rtol=1e-3, atol=1e-5)
