import jax
import numpy as np
import pytest
from noether_optics.optics import propagation, relative_l2
from noether_optics.latency import statistics


@pytest.mark.parametrize("dtype,tol", [("float32", 1e-5), ("float64", 1e-11)])
@pytest.mark.parametrize("n,nw", [(32, 1), (64, 3), (96, 5)])
def test_complex_field_power_dtype_and_dynamic_input(dtype, tol, n, nw):
    rng = np.random.default_rng(30)
    x = (rng.normal(size=(n, n))*4e-8).astype(dtype)
    w = np.linspace(6e-7, 8e-7, nw).astype(dtype)
    a = np.asarray(jax.jit(propagation(n, dtype, "dlux"))(x, w))
    b = np.asarray(jax.jit(propagation(n, dtype, "jax"))(x, w))
    assert relative_l2(a, b) < tol
    assert a.dtype == (np.complex64 if dtype == "float32" else np.complex128)
    np.testing.assert_allclose(np.sum(abs(a)**2, axis=(-1,-2)), 1, rtol=tol)
    assert not np.allclose(a, propagation(n, dtype, "dlux")(x * 2, w))


def test_quantiles_and_strict_deadline_definition():
    s = statistics(np.array([0., .0005, .001, .002]))
    assert s["p50_ms"] == .75
    assert s["max_ms"] == 2
    assert s["miss_1ms"] == .25
    assert s["miss_0.5ms"] == .5
    with pytest.raises(ValueError):
        statistics(np.array([np.nan]))
