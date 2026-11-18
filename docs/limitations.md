# Limitations

- No demonstrated live profitability, broker-fill validation, learned-model generalization or autonomous production deployment.
- Source-only component extraction, not an operational feed-to-CUDA application. Trained weights, fitted transforms, private adapters and data are excluded.
- Original V4 inventory-aware simulation differs from the private flat-owner SHADOW contract. Exact original-historical-to-native learned-model parity and immutable historical split/checkpoint-selection provenance remain incomplete.
- Only the hash-locked Linux/CPython3.11 CPU profile is freshly installation-tested and advisory-scanned. No CUDA test or complete native dependency vulnerability certification is claimed. Existing external native dependencies require maintained versions and local requalification.
- Array order, target meaning, precision, batch size and hardware are part of the numerical contract. Matching shape and synthetic tolerances do not certify arbitrary model artifacts.
- Three inherited Torch SDPA deprecation warnings remain; the attention backend selection and all model/learning equations are unchanged.
- Three inherited candidate boundary defects were corrected, and unused profiler/export/scanner code was removed. The candidate is intentionally not byte-identical to the old deployment; every deviation is recorded in source_map.json.
- Personal contribution is owner-attested. AI-assisted C++ implementation and established-method attribution remain explicit; no independent authorship certification is implied.
- Source-only publication excludes private performance figures as well as market-data payloads. Those private results need their own disclosure decision; no vendor/data redistribution right is inferred.

- The signed-Beta log-density has a confirmed affine-Jacobian sign discrepancy. Existing finite-update tests do not establish correct entropy normalization; [the numerical contract](numerical_contract.md#signed-beta-log-density-known-issue) records the uniform-case check and scope. No algorithm or weight change is included in this documentation revision.
