"""Shared conservative envelopes for deterministic numeric evidence archives."""
from __future__ import annotations


MIB = 1024**2
GIB = 1024**3
FLOAT64_BYTES = 8
INT64_BYTES = 8
BOOL_BYTES = 1

# Numeric evidence is DEFLATE-compressed one member at a time. Compression is
# data-dependent, so admission takes no compression credit. Twice the raw payload
# plus 4 KiB per member is a deliberately loose allowance for NPY headers,
# DEFLATE expansion, ZIP64 records and directory entries.
ARCHIVE_PAYLOAD_FACTOR = 2
ARCHIVE_MEMBER_ALLOWANCE_BYTES = 4096
METADATA_MINIMUM_BYTES = 512 * MIB
WRITE_HEADROOM_MINIMUM_BYTES = 512 * MIB
OPERATING_RESERVE_MINIMUM_BYTES = 4 * GIB


def positive_integer(value, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def round_up(value: int, quantum: int = MIB) -> int:
    return ((value + quantum - 1) // quantum) * quantum


def archive_category(
    name: str,
    count: int,
    payload_per_archive: int,
    members_per_archive: int,
) -> dict:
    """Describe one homogeneous group of numeric archives without writing it."""
    count = positive_integer(count, name + " archive count")
    payload_per_archive = positive_integer(payload_per_archive, name + " payload")
    members_per_archive = positive_integer(members_per_archive, name + " member count")
    numeric = count * payload_per_archive
    per_archive = (
        ARCHIVE_PAYLOAD_FACTOR * payload_per_archive
        + ARCHIVE_MEMBER_ALLOWANCE_BYTES * members_per_archive
    )
    return {
        "name": name,
        "archives": count,
        "members_per_archive": members_per_archive,
        "numeric_payload_bytes": numeric,
        "archive_envelope_bytes": count * per_archive,
        "largest_archive_envelope_bytes": per_archive,
    }


def finalize_archive_budget(categories: list[dict]) -> dict[str, int]:
    """Add metadata, write headroom and operating reserve to archive groups."""
    if not isinstance(categories, list) or not categories:
        raise ValueError("Resource sizing requires at least one archive category")
    numeric = sum(row["numeric_payload_bytes"] for row in categories)
    archives = sum(row["archive_envelope_bytes"] for row in categories)
    largest = max(row["largest_archive_envelope_bytes"] for row in categories)
    # Canonical JSON repeats immutable references by design. Reserve another one
    # eighth of archive envelopes, with a 512 MiB floor, for manifests, hashes,
    # terminal records, diagnostics and filesystem allocation.
    metadata = max(METADATA_MINIMUM_BYTES, round_up((archives + 7) // 8))
    retained = archives + metadata
    write_headroom = max(WRITE_HEADROOM_MINIMUM_BYTES, round_up(largest))
    operating_reserve = max(
        OPERATING_RESERVE_MINIMUM_BYTES,
        round_up((retained + 9) // 10),
    )
    return {
        "numeric_payload_bytes": numeric,
        "archive_envelope_bytes": archives,
        "metadata_filesystem_envelope_bytes": metadata,
        "retained_evidence_envelope_bytes": retained,
        "largest_single_archive_envelope_bytes": largest,
        "temporary_write_headroom_bytes": write_headroom,
        "operating_reserve_bytes": operating_reserve,
        "required_free_bytes": retained + write_headroom + operating_reserve,
        "compression_credit_bytes": 0,
    }
