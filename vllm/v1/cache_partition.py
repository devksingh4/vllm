# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-model KV cache partitioning. ``cache_partition_id`` identifies a
logical owner for KV block ref accounting; see ``BlockPool`` for optional
per-partition ref caps and cost-aware eviction."""

CACHE_PARTITION_ID_EXTRA_ARG: str = "cache_partition_id"

# Used for scheduler-internal block pool operations where refs are not tied
# to a user request partition.
KV_CACHE_INTERNAL_PARTITION_ID: str = "__vllm_kv_internal__"
