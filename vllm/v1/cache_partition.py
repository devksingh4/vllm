# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-model KV cache partitioning (plumbing and metrics).

`cache_partition_id` identifies a logical owner for KV block ref accounting
(e.g. a served base model or an explicit tenant). Scheduling and eviction
policies are unchanged; only attribution for metrics is tracked today.

Request metadata:
    - Generation: ``SamplingParams.extra_args["cache_partition_id"]`` (string).
    - Pooling: ``PoolingParams.extra_kwargs["cache_partition_id"]`` (string).

If unset, the engine falls back to the served model name (same resolution as
``get_served_model_name`` in ``vllm.config.model``).
"""

# Key in SamplingParams.extra_args / PoolingParams.extra_kwargs.
CACHE_PARTITION_ID_EXTRA_ARG: str = "cache_partition_id"

# Used for scheduler-internal GPU/CPU block pool operations (e.g. offload
# bookkeeping) where refs are not tied to a user request partition.
KV_CACHE_INTERNAL_PARTITION_ID: str = "__vllm_kv_internal__"
