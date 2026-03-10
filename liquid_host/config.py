"""Model registry and configuration."""

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "liquid-host" / "models"


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a downloadable Liquid AI model."""

    repo_id: str
    name: str
    family: str
    params: str
    active_params: str
    architecture: str
    description: str
    recommended_dtype: str = "bfloat16"
    default_max_tokens: int = 512
    tags: tuple[str, ...] = ()


# All available Liquid AI models on HuggingFace
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # --- LFM2 family ---
    "lfm2-350m": ModelSpec(
        repo_id="LiquidAI/LFM2-350M",
        name="LFM2-350M",
        family="lfm2",
        params="350M",
        active_params="350M",
        architecture="dense-hybrid",
        description="Ultra-edge dense hybrid model",
        tags=("edge", "tiny"),
    ),
    "lfm2-700m": ModelSpec(
        repo_id="LiquidAI/LFM2-700M",
        name="LFM2-700M",
        family="lfm2",
        params="700M",
        active_params="700M",
        architecture="dense-hybrid",
        description="Edge-class dense hybrid model",
        tags=("edge",),
    ),
    "lfm2-1.2b": ModelSpec(
        repo_id="LiquidAI/LFM2-1.2B",
        name="LFM2-1.2B",
        family="lfm2",
        params="1.2B",
        active_params="1.2B",
        architecture="dense-hybrid",
        description="Dense hybrid (10 conv + 6 attn layers)",
        tags=("small",),
    ),
    "lfm2-2.6b": ModelSpec(
        repo_id="LiquidAI/LFM2-2.6B",
        name="LFM2-2.6B",
        family="lfm2",
        params="2.6B",
        active_params="2.6B",
        architecture="dense-hybrid",
        description="Mid-range dense hybrid model",
        tags=("medium",),
    ),
    "lfm2-2.6b-exp": ModelSpec(
        repo_id="LiquidAI/LFM2-2.6B-Exp",
        name="LFM2-2.6B-Exp",
        family="lfm2",
        params="2.6B",
        active_params="2.6B",
        architecture="dense-hybrid",
        description="Pure RL-trained experimental variant",
        tags=("medium", "experimental"),
    ),
    "lfm2-8b-a1b": ModelSpec(
        repo_id="LiquidAI/LFM2-8B-A1B",
        name="LFM2-8B-A1B",
        family="lfm2",
        params="8B",
        active_params="1B",
        architecture="moe",
        description="MoE model (8B total, 1B active per token)",
        tags=("moe",),
    ),
    "lfm2-24b-a2b": ModelSpec(
        repo_id="LiquidAI/LFM2-24B-A2B",
        name="LFM2-24B-A2B",
        family="lfm2",
        params="24B",
        active_params="2.3B",
        architecture="moe",
        description="Large MoE model (24B total, 2.3B active, 30 conv + 10 attn)",
        tags=("moe", "large"),
    ),
    # --- LFM2.5 family ---
    "lfm2.5-1.2b-base": ModelSpec(
        repo_id="LiquidAI/LFM2.5-1.2B-Base",
        name="LFM2.5-1.2B-Base",
        family="lfm2.5",
        params="1.2B",
        active_params="1.2B",
        architecture="dense-hybrid",
        description="Pre-trained base model (28T tokens)",
        tags=("base",),
    ),
    "lfm2.5-1.2b-instruct": ModelSpec(
        repo_id="LiquidAI/LFM2.5-1.2B-Instruct",
        name="LFM2.5-1.2B-Instruct",
        family="lfm2.5",
        params="1.2B",
        active_params="1.2B",
        architecture="dense-hybrid",
        description="Instruction-tuned model (most popular)",
        tags=("instruct", "recommended"),
    ),
    "lfm2.5-1.2b-thinking": ModelSpec(
        repo_id="LiquidAI/LFM2.5-1.2B-Thinking",
        name="LFM2.5-1.2B-Thinking",
        family="lfm2.5",
        params="1.2B",
        active_params="1.2B",
        architecture="dense-hybrid",
        description="Reasoning/chain-of-thought variant",
        tags=("reasoning",),
    ),
    "lfm2.5-1.2b-jp": ModelSpec(
        repo_id="LiquidAI/LFM2.5-1.2B-JP",
        name="LFM2.5-1.2B-JP",
        family="lfm2.5",
        params="1.2B",
        active_params="1.2B",
        architecture="dense-hybrid",
        description="Japanese-optimized variant",
        tags=("japanese",),
    ),
    "lfm2.5-vl-1.6b": ModelSpec(
        repo_id="LiquidAI/LFM2.5-VL-1.6B",
        name="LFM2.5-VL-1.6B",
        family="lfm2.5",
        params="1.6B",
        active_params="1.6B",
        architecture="dense-hybrid",
        description="Vision-language model",
        tags=("multimodal", "vision"),
    ),
    "lfm2.5-audio-1.5b": ModelSpec(
        repo_id="LiquidAI/LFM2.5-Audio-1.5B",
        name="LFM2.5-Audio-1.5B",
        family="lfm2.5",
        params="1.5B",
        active_params="1.5B",
        architecture="dense-hybrid",
        description="Audio-language model",
        tags=("multimodal", "audio"),
    ),
}


@dataclass
class GenerationConfig:
    """Default generation parameters for Liquid AI models."""

    max_new_tokens: int = 512
    temperature: float = 0.3
    min_p: float = 0.15
    repetition_penalty: float = 1.05
    do_sample: bool = True
    top_p: float = 1.0
    top_k: int = 0

    def to_dict(self) -> dict:
        d = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
        }
        if self.min_p > 0:
            d["min_p"] = self.min_p
        if self.top_p < 1.0:
            d["top_p"] = self.top_p
        if self.top_k > 0:
            d["top_k"] = self.top_k
        return d


def resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve the model cache directory."""
    path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_spec(model_key: str) -> ModelSpec:
    """Look up a model spec by key (case-insensitive)."""
    key = model_key.lower().strip()
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]
    # Try matching by repo_id suffix
    for spec in MODEL_REGISTRY.values():
        if spec.repo_id.lower().endswith(key):
            return spec
    raise KeyError(
        f"Unknown model '{model_key}'. Available: {', '.join(MODEL_REGISTRY.keys())}"
    )
