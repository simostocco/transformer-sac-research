# Third-party notices

The native NumPy-compatible float-log reduction/coefficient design and pairwise sum behavior derive from NumPy 2.2.6. Its full BSD-3-Clause license is retained in licenses/NUMPY-BSD-3-Clause.txt. Upstream [algorithm](https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/loops_exponent_log.dispatch.c.src), [coefficients](https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/npy_simd_data.h), [license](https://github.com/numpy/numpy/blob/v2.2.6/LICENSE.txt).

The requirements lock references external wheels; no wheel or dependency source is vendored. The tested Python dependencies use their own BSD/MIT/Apache/PSF/dual notices; dependency inventory and artifact hashes are recorded in docs/dependencies.json and the lock. No additional license grant over third-party material is implied.

| External dependency | Role / license boundary |
|---|---|
| NumPy / pandas | Arrays/features; BSD-3-Clause and distribution notices |
| PyTorch | Model/SAC execution; BSD-3-Clause plus distribution notices |
| pytest | Synthetic tests; MIT |
| PyArrow | Optional user-owned Parquet input; Apache-2.0 |
| Intel oneMKL | CBLAS header/runtime for native numerical family; installed Intel component terms, not vendored or relicensed |
| OpenSSL | SHA256 admission integrity; checked3.x distribution Apache-2.0, external |
| Gitleaks | Private audit tool, MIT; not a project runtime dependency |

[Intel's oneMKL FAQ](https://www.intel.com/content/www/us/en/developer/articles/tool/onemkl-license-faq.html) points to the actual component license. No SDK, data, trained weights or native dependency binaries are distributed. Installed wheel/native distribution terms must be followed if anyone later bundles binaries; this source-only review is not that redistribution clearance.

Established methods are credited: [SAC](https://arxiv.org/abs/1801.01290), [RoPE](https://arxiv.org/abs/2104.09864), [SimBa](https://arxiv.org/abs/2410.09754).
