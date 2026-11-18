# Clean repository boundary

This source-only allowlist is the entire proposed repository tree. Everything else is excluded, including .git from the private archive, review material, raw logs, recordings, real feature tables, symbol-level results, private benchmark figures, models/scalers/centers, credentials and vendor headers/libraries.

The original isolated preparation branch has private ancestry containing a credential, restricted vendor headers and market-derived tables. It is not a public-release base. The proposed clean local repository is created from this allowlist with a new root and no remotes, parent commits, alternates or borrowed objects. The 60 unpushed operational commits and all original history stay intact and private. Never change visibility of the operational repository or publish its branches.

A clean source snapshot prevents the historical Polygon literal from entering this repository; it does not prove provider-side key revocation. The owner reports the key is probably disabled, so revocation is UNKNOWN and should be confirmed separately without reproducing its value. No provider request or rotation was performed.

No Nanex SDK implementation/header/binary, private scanner strategy or market-data payload is included. The project-side abstract interface and synthetic mock are sufficient for native tests. Source ownership is owner-attested; NumPy attribution and dependency notices remain separate. Excluding data/results removes the need to redistribute them with this source release; it does not clear a separate data/benchmark publication.

The applicable vendor/venue agreements are unavailable for independent clearance. Private operational measurements and ETH application material therefore remain outside this repository. Publicly accessible API documentation or derived-data status is not permission to redistribute actual data. The owner approved MIT for the selected project code; final publication remains unapproved as recorded in the private OWNER_DECISION.md.

No push, PR submission, merge, visibility change or publication is part of the preparation. The local branch and draft PR are review artifacts only.
