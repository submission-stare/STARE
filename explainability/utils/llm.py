"""
LLM Provider – multi-provider registry supporting OpenRouter, Ollama, and AWS Bedrock.

Provider selection (for the default ``get_llm()`` path):
  Set ``LLM_PROVIDER`` env var to ``"openrouter"`` (default), ``"ollama"``, or ``"bedrock"``.

Multi-provider usage (for the Hypothesis Forum):
  Panelists are defined in ``config/settings.yaml`` under ``forum.panelists``
  as a list of ``provider:model`` strings.  Override with the ``FORUM_PANELISTS``
  env var (comma-separated).

  Call ``get_forum_panelists()`` to get a list of ``(label, ChatModel)`` tuples.
  Call ``get_llm_by_provider(provider, model, ...)`` to instantiate any single provider.

Two tiers (for ``get_llm()`` only):
  • LIGHT  – cheap model for high-volume, small-context prompts.
  • HEAVY  – high-quality model for deeper reasoning.

Temperature guidelines for analytical workloads:
  • Routing / structured JSON  →  0.0  (deterministic)
  • Code generation            →  0.0  (deterministic)
  • Hypothesis generation      →  0.4  (some creativity, but grounded)
  • Report writing             →  0.2  (fluent yet faithful)

Max-token defaults are set conservatively to avoid runaway costs.
"""

import os
import logging
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler

load_dotenv()

from utils.config import cfg

logger = logging.getLogger(__name__)


def invoke_llm_with_retries(llm, messages, *, attempts: int = 3, base_delay: float = 2.0):
    """Invoke an LLM and retry transient provider/client parse failures."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            logger.warning(
                "LLM invoke failed on attempt %s/%s (%s: %s). Retrying...",
                attempt,
                attempts,
                type(exc).__name__,
                exc,
            )
            time.sleep(base_delay * attempt)
    assert last_exc is not None
    raise last_exc

# ── Default provider selection ──────────────────────────────────────
_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()

# ── OpenRouter models ──────────────────────────────────────────────
_LIGHT_MODEL = os.getenv(
    "OPENROUTER_MODEL_LIGHT", "anthropic/claude-sonnet-4.6"
)
_HEAVY_MODEL = os.getenv(
    "OPENROUTER_MODEL_HEAVY", "anthropic/claude-sonnet-4.6"
)

# ── Ollama config ──────────────────────────────────────────────────
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://98.83.51.182:11434")
_OLLAMA_LIGHT_MODEL = os.getenv("OLLAMA_MODEL_LIGHT", "deepseek-r1:32b")
_OLLAMA_HEAVY_MODEL = os.getenv("OLLAMA_MODEL_HEAVY", "deepseek-r1:32b")

# ── AWS Bedrock config ─────────────────────────────────────────────
_BEDROCK_REGION = os.getenv("AWS_BEDROCK_REGION", "us-east-1")
_BEDROCK_LIGHT_MODEL = os.getenv("BEDROCK_MODEL_LIGHT", "us.anthropic.claude-sonnet-4-20250514-v1:0")
_BEDROCK_HEAVY_MODEL = os.getenv("BEDROCK_MODEL_HEAVY", "us.anthropic.claude-sonnet-4-20250514-v1:0")

# ── Max tokens for the *response* (not prompt) ────────────────────
_DEFAULT_MAX_TOKENS_LIGHT = int(os.getenv("LLM_MAX_TOKENS_LIGHT", "2048"))
_DEFAULT_MAX_TOKENS_HEAVY = int(os.getenv("LLM_MAX_TOKENS_HEAVY", "4096"))

# ── Forum panelists (resolved via cfg → YAML + env override) ─────


# ═══════════════════════════════════════════════════════════════════
# Low-level factory: create an LLM instance for a specific provider
# ═══════════════════════════════════════════════════════════════════

def get_llm_by_provider(
    provider: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
):
    """
    Instantiate a LangChain chat model for a specific provider.

    Parameters
    ----------
    provider : ``"openrouter"`` | ``"ollama"`` | ``"bedrock"``
    model : str
        The model identifier (e.g. ``"qwen2.5:14b"``, ``"anthropic/claude-sonnet-4"``).
    temperature : float
    max_tokens : int
    """
    provider = provider.lower().strip()

    if provider == "ollama":
        base_url = _OLLAMA_BASE_URL
        logger.info(
            f"LLM init [ollama]: model={model}, "
            f"base_url={base_url}, temp={temperature}, max_tokens={max_tokens}"
        )
        return ChatOpenAI(
            model=model,
            openai_api_key="ollama",
            openai_api_base=f"{base_url}/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

    elif provider == "bedrock":
        try:
            from langchain_aws import ChatBedrockConverse
        except ImportError:
            raise ImportError(
                "langchain-aws is required for Bedrock support. "
                "Install it with: pip install langchain-aws"
            )
        logger.info(
            f"LLM init [bedrock]: model={model}, "
            f"region={_BEDROCK_REGION}, temp={temperature}, max_tokens={max_tokens}"
        )
        return ChatBedrockConverse(
            model=model,
            region_name=_BEDROCK_REGION,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
        logger.info(
            f"LLM init [openrouter]: model={model}, "
            f"temp={temperature}, max_tokens={max_tokens}"
        )
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Use 'openrouter', 'ollama', or 'bedrock'.")


# ═══════════════════════════════════════════════════════════════════
# Default get_llm() – backwards-compatible, uses LLM_PROVIDER env
# ═══════════════════════════════════════════════════════════════════

def get_llm(
    temperature: float = 0.0,
    tier: str = "light",
    max_tokens: int | None = None,
):
    """
    Return a LangChain chat model instance using the default provider (``LLM_PROVIDER``).

    This is the original interface — all existing nodes use this function unchanged.

    Parameters
    ----------
    temperature : float
        Sampling temperature.  0.0 = deterministic.
    tier : ``"light"`` | ``"heavy"``
        Which model tier to use.
    max_tokens : int, optional
        Override the default max-token cap for the response.
    """
    if tier == "heavy":
        default_max = _DEFAULT_MAX_TOKENS_HEAVY
    else:
        default_max = _DEFAULT_MAX_TOKENS_LIGHT

    resolved_max = max_tokens or default_max

    if _LLM_PROVIDER == "ollama":
        model = _OLLAMA_HEAVY_MODEL if tier == "heavy" else _OLLAMA_LIGHT_MODEL
    elif _LLM_PROVIDER == "bedrock":
        model = _BEDROCK_HEAVY_MODEL if tier == "heavy" else _BEDROCK_LIGHT_MODEL
    else:
        model = _HEAVY_MODEL if tier == "heavy" else _LIGHT_MODEL

    return get_llm_by_provider(
        provider=_LLM_PROVIDER,
        model=model,
        temperature=temperature,
        max_tokens=resolved_max,
    )


# ═══════════════════════════════════════════════════════════════════
# Forum panelists – multiple providers instantiated simultaneously
# ═══════════════════════════════════════════════════════════════════

def _parse_panelist_spec(spec: str) -> tuple[str, str]:
    """
    Parse a panelist spec string like ``"ollama:qwen2.5:14b"`` or
    ``"openrouter:anthropic/claude-sonnet-4"`` into ``(provider, model)``.

    The first token before ':' is the provider; the rest is the model name
    (which may itself contain ':'  — e.g. Ollama tags ``qwen2.5:14b``).
    """
    spec = spec.strip()
    if ":" not in spec:
        raise ValueError(f"Invalid panelist spec '{spec}'. Expected 'provider:model'.")
    provider, model = spec.split(":", 1)
    return provider.strip(), model.strip()


def get_forum_panelists(
    temperature: float = 0.4,
    max_tokens: int | None = None,
) -> list[tuple[str, object]]:
    """
    Return a list of ``(label, ChatModel)`` tuples from the panelist config.

    Resolution order (handled by ``cfg()``):
      1. ``FORUM_PANELISTS`` env var  (comma-separated specs)
      2. ``forum.panelists`` list in ``config/settings.yaml``
      3. Fallback: single panelist using the default provider.

    Each panelist is an independent LLM instance that can generate hypotheses
    concurrently.  Labels are auto-generated as ``"provider/model"``.

    Parameters
    ----------
    temperature : float
        Shared temperature for all panelists.
    max_tokens : int, optional
        Shared max-token cap.  Defaults to ``_DEFAULT_MAX_TOKENS_LIGHT``.

    Returns
    -------
    list[tuple[str, ChatModel]]
    """
    resolved_max = max_tokens or _DEFAULT_MAX_TOKENS_LIGHT

    panelist_specs: list[str] = cfg("forum.panelists", [])

    if not panelist_specs:
        # Fallback: just the default provider
        label = f"{_LLM_PROVIDER}/{_OLLAMA_LIGHT_MODEL if _LLM_PROVIDER == 'ollama' else _LIGHT_MODEL}"
        return [(label, get_llm(temperature=temperature, max_tokens=resolved_max))]

    panelists = []
    for spec in panelist_specs:
        spec = spec.strip()
        if not spec:
            continue
        try:
            provider, model = _parse_panelist_spec(spec)
            label = f"{provider}/{model}"
            llm = get_llm_by_provider(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=resolved_max,
            )
            panelists.append((label, llm))
            logger.info(f"Forum panelist registered: {label}")
        except Exception as e:
            logger.error(f"Failed to create panelist from spec '{spec}': {e}")

    if not panelists:
        logger.warning("No valid forum panelists configured. Falling back to default provider.")
        label = f"{_LLM_PROVIDER}/default"
        return [(label, get_llm(temperature=temperature, max_tokens=resolved_max))]

    return panelists
