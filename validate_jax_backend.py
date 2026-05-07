"""
One-time validation that the JAX Abeles kernel matches the C backend.

Run once during development before using the GPU optimizer. Results are
printed; no return value. Raises AssertionError if any test fails.

Usage:
    python validate_jax_backend.py
    # or inside a notebook:
    exec(open('/homes/dfs1/Refltools/validate_jax_backend.py').read())
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from refnx.reflect._jax_reflect import abeles_jax
from refnx.reflect import _creflect as _c

TOL = 1e-10  # maximum allowed absolute difference


def _make_layers(n_inner_layers, seed=0):
    """Return a (n_inner_layers+2, 4) slab array with random realistic values."""
    rng = np.random.default_rng(seed)
    layers = np.zeros((n_inner_layers + 2, 4), dtype=np.float64)
    # fronting: Si-like, no thickness
    layers[0] = [0.0, 2.07, 0.0, 0.0]
    # backing: air
    layers[-1] = [0.0, 0.0, 0.0, 3.0 + rng.random() * 5]
    # inner layers: random but physically plausible
    for i in range(1, n_inner_layers + 1):
        layers[i] = [
            5.0 + rng.random() * 100,    # thickness 5–105 Å
            rng.random() * 6 - 1,         # SLD -1 to 5 ×10⁻⁶ Å⁻²
            rng.random() * 0.5,           # iSLD 0–0.5 ×10⁻⁶ Å⁻²
            2.0 + rng.random() * 8,       # roughness 2–10 Å
        ]
    return layers


def _compare(q, layers, scale=1.0, bkg=1e-7, label=""):
    c_result = _c.abeles(q, layers, scale=scale, bkg=bkg)
    jax_result = np.array(abeles_jax(jnp.array(q), jnp.array(layers), scale=scale, bkg=bkg))
    max_diff = np.max(np.abs(c_result - jax_result))
    rel_diff = np.max(np.abs(c_result - jax_result) / (np.abs(c_result) + 1e-30))
    ok = max_diff < TOL
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: max_abs={max_diff:.2e}  max_rel={rel_diff:.2e}")
    assert ok, f"JAX/C mismatch ({label}): max absolute diff = {max_diff:.2e} exceeds {TOL:.0e}"
    return c_result, jax_result


def run_validation():
    print("=" * 60)
    print("JAX vs C backend numerical validation")
    print(f"  GPU: {jax.devices()}")
    print(f"  Tolerance: {TOL:.0e}")
    print("=" * 60)

    q_fine = np.linspace(0.005, 0.3, 500, dtype=np.float64)
    q_coarse = np.linspace(0.01, 0.25, 80, dtype=np.float64)
    q_xray = np.linspace(0.001, 0.15, 300, dtype=np.float64)  # soft X-ray range

    print("\n1. Single layer (substrate/film/air):")
    _compare(q_fine, _make_layers(1, seed=0), label="n_layers=1, q_fine")
    _compare(q_fine, _make_layers(1, seed=1), label="n_layers=1, q_fine (seed 1)")

    print("\n2. Multi-layer stacks:")
    for n in [3, 5, 8, 12]:
        _compare(q_fine, _make_layers(n, seed=n), label=f"n_layers={n}")

    print("\n3. Soft X-ray Q range (low Q):")
    _compare(q_xray, _make_layers(4, seed=42), label="n_layers=4, soft X-ray Q")
    _compare(q_coarse, _make_layers(6, seed=99), label="n_layers=6, coarse Q")

    print("\n4. Scale and background variations:")
    layers = _make_layers(4, seed=7)
    _compare(q_fine, layers, scale=0.95, bkg=1e-8, label="scale=0.95 bkg=1e-8")
    _compare(q_fine, layers, scale=1.05, bkg=5e-6, label="scale=1.05 bkg=5e-6")

    print("\n5. High absorption (large iSLD):")
    layers_absorb = _make_layers(3, seed=13)
    layers_absorb[1:4, 2] = [2.0, 3.5, 1.0]  # strong absorption
    _compare(q_fine, layers_absorb, label="high iSLD")

    print("\n6. Batched vmap consistency:")
    from jax import vmap
    n_batch = 20
    batch_layers = np.stack([_make_layers(4, seed=i) for i in range(n_batch)])
    batched_fn = jit(vmap(abeles_jax, in_axes=(None, 0)))
    jax_batch = np.array(batched_fn(jnp.array(q_fine), jnp.array(batch_layers)))
    for i in range(n_batch):
        c_i = _c.abeles(q_fine, batch_layers[i])
        max_diff = np.max(np.abs(c_i - jax_batch[i]))
        assert max_diff < TOL, f"Batch item {i}: diff = {max_diff:.2e}"
    print(f"  [PASS] vmap over {n_batch} layer sets, max_abs<{TOL:.0e}")

    print("\n" + "=" * 60)
    print("All validation tests PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    from jax import jit
    run_validation()
