# Reproducibility

Supported tested profile: Linux x86-64 / CPython 3.11 / CPU. The hash lock pins all 23 third-party distributions, including pip, setuptools and wheel. Torch's official CPU wheel URL is explicit, so installation does not silently resolve the CUDA distribution. NumPy 2.2.6 is retained for the qualified numerical family. No model or market data download occurs in tests.

From the repository root:

```bash
python3.11 -m venv /tmp/transformer-sac-research
/tmp/transformer-sac-research/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements/linux-cpu-py311.lock
/tmp/transformer-sac-research/bin/python -m pip install --no-deps --no-build-isolation .
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/transformer-sac-research/bin/python -m pytest -q -p no:cacheprovider
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /tmp/transformer-sac-research/bin/python examples/synthetic_cpu.py
/tmp/transformer-sac-research/bin/python -m pip check
```

Installing the wheel instead of editable source is tested separately with pytest's source pythonpath disabled. Broad pyproject bounds are compatibility metadata; only the locked profile has an advisory scan and fresh-install evidence. Other Python/platform versions require their own wheel hashes and validation. Network is needed to install packages; the research checks themselves are offline.

## Native controls

Native prerequisites: C++20 compiler, CMake >=3.20, OpenSSL development headers/libcrypto, Intel oneMKL CBLAS header/runtime, pthreads and dl. Dependencies are externally installed, not vendored. A clean host lacking OpenSSL fails at configure; supplying OpenSSL alone does not supply oneMKL. The final native controls use isolated OpenSSL 3.5.9 and external oneMKL 2022.1; they are evidence versions, not a public binary bundle or full native supply-chain security certification. The candidate uses libcrypto for SHA256 integrity, not network/TLS/ASN.1 processing. OpenSSL 3.5.9 was the latest binary observed in the checked 3.5 channel; upstream 3.5.10 was not yet packaged there. Known advisories cover PEM, TLS, QUIC, CMS and cipher paths; see the [OpenSSL 3.5 advisory list](https://openssl-library.org/news/vulnerabilities-3.5/). The candidate calls only SHA256 APIs, and no affected call path was identified. This is usage triage, not clearance of every OpenSSL operation. Use maintained external libraries and rerun the controls after upgrades.

```bash
cmake -S . -B /tmp/transformer-sac-native -DCMAKE_BUILD_TYPE=Debug \
  -DOPENSSL_ROOT_DIR="$RESEARCH_OPENSSL_PREFIX" \
  -DMKL_INCLUDE_DIR="$RESEARCH_MKL_PREFIX/include" \
  -DMKL_RT_LIBRARY="$RESEARCH_MKL_PREFIX/lib/libmkl_rt.so"
cmake --build /tmp/transformer-sac-native -j2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ctest --test-dir /tmp/transformer-sac-native --output-on-failure
RESEARCH_NATIVE_BUILD=/tmp/transformer-sac-native CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /tmp/transformer-sac-research/bin/python -m pytest -q -p no:cacheprovider
```

The context suite requires a local UNIX socketpair; a restricted sandbox may reject send with EPERM. Tests passed with host IPC permission. Compiler flags disable fast-math and implicit contraction; explicit FMA in the attributed NumPy-compatible routine remains intentional. A generic BLAS substitution or altered numerical backend needs requalification.

## What cannot be reproduced here

The private trained/live workload additionally requires the original operational adapter, vendor SDK/library and licensed source, frozen Transformer/SAC weights, fitted preprocessing and the historical CUDA batch/precision family. They are excluded. The public package has no checkpoint loader, feed launcher, training CLI or broker execution. Trusted caller-supplied in-memory Python models remain executable code. Synthetic tests support the component contracts, not equivalence to unavailable learned artifacts, profitability or complete historical parity.
