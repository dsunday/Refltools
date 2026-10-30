"""Synthetic check: nexafs_batch.fit_one recovers known peaks. Run: python tests/test_nexafs_batch.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nexafs_models as M
import nexafs_batch as B
from NEXAFS_BuildingBlock import gaussian, edge_bl_func

m = B.relaxed(M.SOC_CARBON_UV)
E = np.arange(280, 330, 0.1)
pk = {k: v for k, v in m["peaks"].items() if not k.endswith("_bounds")}
truth = dict(pk, peak_3_height=0.9, peak_5_energy=289.0)
y = edge_bl_func(E, 289.5, 1.43, 0.6, 0.01) + sum(
    gaussian(E, truth[f"peak_{i}_height"], truth[f"peak_{i}_energy"], truth[f"peak_{i}_width"]) for i in range(1, 10))
f = B.fit_one(E, y + np.random.default_rng(0).normal(0, 0.003, E.size), m)
assert f["results"]["r_squared"] > 0.999, f["results"]["r_squared"]
assert abs(f["peaks"].set_index("peak").loc[3, "height"] - 0.9) < 0.1
# isld normalisation round trip
yn, pre, sc = B.normalise_isld(np.array([270., 280., 325., 330.]), np.array([1., 1., 4.2, 4.2]))
assert abs(pre - 1) < 1e-9 and abs(yn[-1] - 1.43) < 1e-9
print("ok")
