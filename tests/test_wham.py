import warnings

import numpy as np
import pytest

from src.potentials import true_free_energy
from src.wham import wham_1d

KT = 0.4
BINS = np.linspace(-1.8, 1.7, 71)


def exact_window_samples(centers, k, n, rng):
    """Independent samples from exp(-beta [F(x) + U_i(x)]) by inverse-CDF on a fine grid."""
    fine = np.linspace(-2.2, 2.1, 20001)
    dx = fine[1] - fine[0]
    out = []
    for c in centers:
        p = np.exp(-(true_free_energy(fine) + 0.5 * k * (fine - c) ** 2) / KT)
        cdf = np.cumsum(p) / p.sum()
        out.append(np.interp(rng.random(n), cdf, fine) + rng.uniform(-dx / 2, dx / 2, n))
    return np.column_stack(out)


def test_wham_recovers_known_profile():
    rng = np.random.default_rng(0)
    centers = np.linspace(-1.5, 1.4, 14)
    samples = exact_window_samples(centers, 40.0, 20000, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a good data set must not trigger warnings
        res = wham_1d(samples, centers, 40.0, KT, BINS)
    assert res["converged"] and res["reliable"]
    Ft = true_free_energy(res["x"])
    region = Ft < 3.3
    assert np.sqrt(np.mean((res["free_energy"][region] - Ft[region]) ** 2)) < 0.1 * KT
    # Normalized density.
    assert np.sum(res["probability"] * np.diff(BINS)) == pytest.approx(1.0)
    # Offsets are gauge-fixed to the first window.
    assert res["offsets"][0] == 0.0


def test_window_order_does_not_matter():
    rng = np.random.default_rng(1)
    centers = np.linspace(-1.5, 1.4, 12)
    samples = exact_window_samples(centers, 40.0, 5000, rng)
    perm = rng.permutation(12)
    a = wham_1d(samples, centers, 40.0, KT, BINS)
    b = wham_1d(samples[:, perm], centers[perm], 40.0, KT, BINS)
    np.testing.assert_allclose(a["free_energy"], b["free_energy"], atol=1e-6, equal_nan=True)


def test_empty_bins_are_nan_not_large_numbers():
    rng = np.random.default_rng(2)
    centers = np.linspace(-1.0, 1.0, 8)
    samples = exact_window_samples(centers, 60.0, 3000, rng)
    res = wham_1d(samples, centers, 60.0, KT, BINS)
    empty = res["counts"].sum(axis=0) == 0
    assert empty.any()
    assert np.all(np.isnan(res["free_energy"][empty]))
    assert np.all(res["probability"][empty] == 0.0)
    assert np.all(np.isfinite(res["free_energy"][~empty]))


def test_poor_overlap_warns_and_is_flagged():
    rng = np.random.default_rng(3)
    centers = np.linspace(-1.5, 1.4, 5)
    samples = exact_window_samples(centers, 300.0, 3000, rng)
    with pytest.warns(RuntimeWarning) as record:
        res = wham_1d(samples, centers, 300.0, KT, BINS)
    messages = " ".join(str(w.message) for w in record)
    assert "overlap" in messages and "disconnected" in messages
    assert not res["reliable"]


def test_nonconvergence_is_reported():
    rng = np.random.default_rng(4)
    centers = np.linspace(-1.5, 1.4, 14)
    samples = exact_window_samples(centers, 40.0, 2000, rng)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        res = wham_1d(samples, centers, 40.0, KT, BINS, max_iter=3)
    assert not res["converged"] and not res["reliable"]
