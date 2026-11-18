# Numerical and temporal contract

The retained native input contract uses feature windows 10/10/10/180, epsilon 1e-12, float64 recurrence, float32 emitted features and 120-token inputs. Frozen centers/scalers must not be refitted during inference. The model configuration and feature ordering are part of the artifact identity. The live policy consumes 65 contexts and 490 values: 106 nonembedding values plus a 384-value embedding. Owner inventory, previous exposure and pending-owner fields were deliberately flat in the private SHADOW diagnostics; this differs from the stateful V4 simulator.

Physical-B20 padding and explicit reduction/mutation order preserve the tested numerical family. They are not interchangeable with arbitrary batch sizes, mixed precision, generic BLAS implementations or a different NumPy version. The native CBLAS call uses the same oneMKL runtime resolved by the historical `libblas.so` on this host. Compiler flags disable fast-math and implicit contraction; explicit FMA inside the NumPy-compatible log routine remains intentional.

`input_log` reproduces the positive-float reduction/polynomial design from NumPy 2.2.6. Its coefficients and the pairwise-sum design are conservatively attributed in THIRD_PARTY_NOTICES and the retained NumPy license. References: [algorithm](https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/loops_exponent_log.dispatch.c.src), [coefficients](https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/npy_simd_data.h).

Fresh public checks cover feature-prefix causality, causal attention, original-versus-fast CPU inference on the same random weights, signed action mapping, accounting, recovery, synthetic scheduling and a 4,096-value synthetic native-log/NumPy comparison with absolute tolerance 2e-6. These are scoped controls, not bitwise equivalence for every input or learned-model parity.

The return-delta local bias loops over its configured K; sequence lengths below K are not supported. The candidate's in-memory constructor rejects such bindings instead of allowing the inherited low-level range error.

Private historical controls demonstrate incomplete learned-model equivalence. Real-data goldens, quantitative deployment/parity results and fitted artifacts are not part of this source-only tree.

Native canonical scheduling and temporal owner fields must retain their original contracts. The original V4 environment's `legacy_same_row` and `previous_completed_bar` modes are distinct, and pending targets act on eligible real-trade bars. Do not infer exchange-to-order latency, true trade fills or stateful portfolio behavior from flat-owner diagnostic outputs.
