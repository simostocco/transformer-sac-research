# Validation scope

The public results are synthetic CPU controls, not financial or live-feed benchmarks. Tests use generated arrays/records and small random models. No real prices, symbols, tapes, fitted assets or checkpoints are shipped.

The original candidate had 66 Python tests and 45 subtests. The current extraction adds six regression cases: four forecast/embedding nonfinite cases across both inference wrappers, one timestamp-tie ordering case through direct and converted records, and one unsupported-cadence case. All six failed before their fixes. The revised suite passes **72 tests plus 45 subtests**, with **4/4 native suites** passing. Exact environment/scope appears in validation_results.json and validation_environment.json.

The native suites cover core/history behavior, bounded allocation, queue controls and context microbatch identity/cache/staleness/drain with a mock worker. Native numerical comparison uses 4,096 synthetic positive values and an absolute 2e-6 tolerance against NumPy. This is not universal bitwise equivalence.

The random-weight example consumes 240 synthetic bars and produces finite outputs. A tiny in-memory SAC update tests the code path; it is not a training experiment or evaluation of a policy's economic performance. Three inherited Torch SDPA deprecation warnings remain, with the original attention selection untouched to avoid an unqualified numerical backend change.

Private historical deployment, source/binary/model pins and decision timing were reviewed separately. Those receipts and figures are not public candidate results and are excluded pending disclosure rights. Full learned-model/CUDA/market-data reproduction is outside the public synthetic contract.

## Interpretation of passing controls

The recorded 72 tests and 45 subtests remain valid evidence for their specific assertions; this documentation update does not change source or tests. A separate uniform-Beta density check exposed the [signed-Beta Jacobian-sign issue](numerical_contract.md#signed-beta-log-density-known-issue), which the existing suite does not cover. A finite SAC update does not establish a correct entropy term, trained-policy quality or a complete numerical audit.
