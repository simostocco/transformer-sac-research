# Security boundary

Use the hash-locked Linux/CPython3.11 CPU profile. Its 23 third-party distributions were queried against OSV on 2026-11-18 with no reported version matches, including pip/setuptools/wheel. This does not prove vulnerability absence or cover native/OS libraries. Recheck advisories and rerun controls when updating the lock.

The source tree contains no checkpoint deserializer or live feed/broker launcher. Caller-owned in-memory Python models remain executable code, not a safe artifact format. Do not add untrusted pickle/checkpoint loading. Input/output validation is scoped to the documented contracts.

Never add API keys, vendor SDK payloads, market recordings, real feature tables, model weights or private history. The exact PUBLICATION_MANIFEST.csv allowlist is the reviewed release boundary; .gitignore alone is not a publication review. No key value belongs in an issue or report. Report suspected secrets privately to the owner and handle provider-side revocation separately.

External OpenSSL/oneMKL components are not distributed here and require maintained local installations and their own security/license review. The native use of libcrypto is SHA256 integrity; no claim of complete native supply-chain certification is made.
