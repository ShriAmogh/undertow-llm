"""
lensllm — LLM Observability & Reliability SDK
==================================================

Public API surface:

    from lensllm import track, configure

    @track(cache=True, retries=3)
    def call_llm(prompt: str) -> str:
        ...  # any LLM provider call goes here
"""

from lensllm.decorator import track, PolicyViolationError, get_last_call_info, CallInfo
from lensllm.config import configure, GatewayConfig, get_config

__all__ = [
    "track",
    "configure",
    "get_config",
    "GatewayConfig",
    "PolicyViolationError",
    "get_last_call_info",
    "CallInfo",
]
__version__ = "0.2.0"
