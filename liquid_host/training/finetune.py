"""LoRA/QLoRA fine-tuning for Liquid AI models.

Supports supervised fine-tuning on chat-format JSONL data using PEFT LoRA
adapters and TRL's SFTTrainer.

Data format (one JSON object per line):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LoraConfig:
    """Configuration for LoRA adapter training."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | None = None
    bias: str = "none"


@dataclass
class TrainingConfig:
    """Configuration for the training run."""

    output_dir: str = "./finetune-output"
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True
    quantize_4bit: bool = False
    seed: int = 42
    eval_split: float = 0.05


def load_chat_dataset(data_path: str | Path):
    """Load a JSONL chat dataset and return a HuggingFace Dataset."""
    from datasets import Dataset

    records = []
    data_path = Path(data_path)

    if data_path.suffix == ".jsonl":
        with open(data_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "messages" not in obj:
                    raise ValueError(f"Line {line_num}: missing 'messages' key")
                records.append(obj)
    elif data_path.suffix == ".json":
        with open(data_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            records = data
        else:
            raise ValueError("JSON file must contain a list of objects")
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix} (use .jsonl or .json)")

    logger.info("Loaded %d training examples from %s", len(records), data_path)
    return Dataset.from_list(records)


def finetune(
    model_key: str,
    data_path: str | Path,
    training_config: TrainingConfig | None = None,
    lora_config: LoraConfig | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Run LoRA/QLoRA fine-tuning on a Liquid AI model.

    Args:
        model_key: Key in MODEL_REGISTRY (e.g. 'lfm2.5-1.2b-instruct').
        data_path: Path to JSONL training data.
        training_config: Training hyperparameters.
        lora_config: LoRA adapter configuration.
        cache_dir: HuggingFace cache directory.

    Returns:
        Path to the saved adapter directory.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    from liquid_host.config import get_model_spec, resolve_cache_dir

    tc = training_config or TrainingConfig()
    lc = lora_config or LoraConfig()
    cache = resolve_cache_dir(cache_dir)

    spec = get_model_spec(model_key)
    logger.info("Fine-tuning %s (%s)", spec.name, spec.repo_id)
    logger.info("  Output: %s", tc.output_dir)
    logger.info("  LoRA rank=%d, alpha=%d, dropout=%.2f", lc.rank, lc.alpha, lc.dropout)
    logger.info("  4-bit quantization: %s", tc.quantize_4bit)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repo_id, cache_dir=str(cache), trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (optionally quantized)
    model_kwargs = {
        "pretrained_model_name_or_path": spec.repo_id,
        "cache_dir": str(cache),
        "torch_dtype": torch.bfloat16 if tc.bf16 else torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if tc.quantize_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.config.use_cache = False

    # Determine target modules
    target_modules = lc.target_modules
    if target_modules is None:
        target_modules = _detect_target_modules(model)
        logger.info("Auto-detected target modules: %s", target_modules)

    # Apply LoRA
    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lc.rank,
        lora_alpha=lc.alpha,
        lora_dropout=lc.dropout,
        target_modules=target_modules,
        bias=lc.bias,
    )
    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable parameters: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # Load dataset
    dataset = load_chat_dataset(data_path)
    if tc.eval_split > 0 and len(dataset) > 10:
        split = dataset.train_test_split(test_size=tc.eval_split, seed=tc.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        logger.info("Train: %d examples, Eval: %d examples", len(train_dataset), len(eval_dataset))
    else:
        train_dataset = dataset
        eval_dataset = None

    # Format messages using the model's chat template
    def format_chat(example):
        messages = example["messages"]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            # Fallback: if template doesn't support 'tool' role, convert to 'user'
            safe_messages = []
            for m in messages:
                if m["role"] == "tool":
                    safe_messages.append({"role": "user", "content": f"[Tool Result]\n{m['content']}"})
                else:
                    safe_messages.append(m)
            text = tokenizer.apply_chat_template(
                safe_messages, tokenize=False, add_generation_prompt=False,
            )
        return {"text": text}

    train_dataset = train_dataset.map(format_chat)
    if eval_dataset:
        eval_dataset = eval_dataset.map(format_chat)

    # Training arguments
    output_dir = Path(tc.output_dir)
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=tc.epochs,
        per_device_train_batch_size=tc.batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate,
        warmup_ratio=tc.warmup_ratio,
        weight_decay=tc.weight_decay,
        logging_steps=tc.logging_steps,
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        fp16=tc.fp16,
        bf16=tc.bf16,
        max_length=tc.max_seq_length,
        seed=tc.seed,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=sft_config,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save adapter
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Adapter saved to %s", adapter_dir)

    # Save training config for reference
    config_out = {
        "base_model": spec.repo_id,
        "model_key": model_key,
        "lora": {"rank": lc.rank, "alpha": lc.alpha, "dropout": lc.dropout, "target_modules": target_modules},
        "training": {
            "epochs": tc.epochs, "batch_size": tc.batch_size,
            "learning_rate": tc.learning_rate, "max_seq_length": tc.max_seq_length,
            "quantize_4bit": tc.quantize_4bit,
        },
    }
    with open(adapter_dir / "finetune_config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    return adapter_dir


@dataclass
class RemoteConfig:
    """Configuration for remote fine-tuning via a custom HuggingFace Space."""

    hf_token: str | None = None
    hf_username: str | None = None
    project_name: str = "liquid-host-finetune"
    backend: str = "l4x1"
    private_repo: bool = True


# Maps CLI-friendly names to HF Space hardware strings
_HW_MAP = {
    "t4-small": "t4-small",
    "t4-medium": "t4-medium",
    "a10g-small": "a10g-small",
    "a10g-large": "a10g-large",
    "l4x1": "l4x1",
    "l4x4": "l4x4",
    "l40sx1": "l40s-1x",
    "a100-large": "a100-large",
}


def _build_training_script(
    repo_id: str,
    dataset_repo: str,
    adapter_repo: str,
    tc: TrainingConfig,
    lc: LoraConfig,
) -> str:
    """Generate a self-contained training script to run inside the HF Space."""
    target_modules_str = repr(lc.target_modules) if lc.target_modules else "None"
    quantize_block = ""
    if tc.quantize_4bit:
        quantize_block = '''
    from transformers import BitsAndBytesConfig
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )'''

    return f'''"""Remote LoRA fine-tuning for Liquid AI models on HF Spaces."""

import json
import os
import sys
import logging
import threading
import torch
import torch.nn as nn
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Keep-alive HTTP server so HF Spaces doesn't kill the container
_status = {{"phase": "initializing"}}

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(_status).encode())
    def log_message(self, *args):
        pass  # suppress request logs

def _start_health_server(port=7860):
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info("Health server started on port %d", port)

HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "{repo_id}"
DATASET_REPO = "{dataset_repo}"
ADAPTER_REPO = "{adapter_repo}"

# Training config
EPOCHS = {tc.epochs}
BATCH_SIZE = {tc.batch_size}
GRAD_ACCUM = {tc.gradient_accumulation_steps}
LR = {tc.learning_rate}
WARMUP_RATIO = {tc.warmup_ratio}
WEIGHT_DECAY = {tc.weight_decay}
MAX_SEQ_LEN = {tc.max_seq_length}
SEED = {tc.seed}
BF16 = {tc.bf16}
FP16 = {tc.fp16}

# LoRA config
LORA_RANK = {lc.rank}
LORA_ALPHA = {lc.alpha}
LORA_DROPOUT = {lc.dropout}
TARGET_MODULES = {target_modules_str}
LORA_BIAS = "{lc.bias}"


def detect_target_modules(model):
    target = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            short = name.split(".")[-1]
            target.add(short)
    target -= {{"lm_head", "embed_tokens", "wte", "wpe"}}
    return sorted(target)


def main():
    # Use a writable cache directory (HF Space containers may not have write access to /.cache)
    os.environ["HF_HOME"] = "/tmp/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"

    _start_health_server()
    _status["phase"] = "loading_model"
    logger.info("Loading tokenizer from %s", REPO_ID)
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from %s", REPO_ID)
    model_kwargs = {{
        "pretrained_model_name_or_path": REPO_ID,
        "torch_dtype": torch.bfloat16 if BF16 else torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
        "token": HF_TOKEN,
    }}
{quantize_block}

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.config.use_cache = False

    target_modules = TARGET_MODULES
    if target_modules is None:
        target_modules = detect_target_modules(model)
        logger.info("Auto-detected target modules: %s", target_modules)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias=LORA_BIAS,
    )
    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable: %s / %s (%.2f%%)", f"{{trainable:,}}", f"{{total:,}}", 100 * trainable / total)

    logger.info("Loading dataset from %s", DATASET_REPO)
    dataset = load_dataset(DATASET_REPO, split="train", token=HF_TOKEN)
    logger.info("Dataset size: %d examples", len(dataset))

    def format_chat(example):
        messages = example["messages"]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            safe = []
            for m in messages:
                if m["role"] == "tool":
                    safe.append({{"role": "user", "content": f"[Tool Result]\\n{{m[\'content\']}}"}})
                else:
                    safe.append(m)
            text = tokenizer.apply_chat_template(safe, tokenize=False, add_generation_prompt=False)
        return {{"text": text}}

    dataset = dataset.map(format_chat)

    sft_config = SFTConfig(
        output_dir="/tmp/output",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=FP16,
        bf16=BF16,
        max_length=MAX_SEQ_LEN,
        seed=SEED,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
    )

    _status["phase"] = "training"
    logger.info("Starting training...")
    trainer.train()

    _status["phase"] = "saving"
    logger.info("Saving adapter...")
    adapter_dir = Path("/tmp/output/adapter")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    config_out = {{
        "base_model": REPO_ID,
        "lora": {{"rank": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target_modules": target_modules}},
        "training": {{"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "max_seq_length": MAX_SEQ_LEN}},
    }}
    with open(adapter_dir / "finetune_config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    _status["phase"] = "pushing"
    logger.info("Pushing adapter to Hub: %s", ADAPTER_REPO)
    api = HfApi(token=HF_TOKEN)
    api.create_repo(ADAPTER_REPO, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=ADAPTER_REPO,
        repo_type="model",
        commit_message="LoRA adapter from liquid-host fine-tuning",
    )
    _status["phase"] = "complete"
    logger.info("Done! Adapter pushed to %s", ADAPTER_REPO)


if __name__ == "__main__":
    main()
'''


def finetune_remote(
    model_key: str,
    data_path: str | Path,
    training_config: TrainingConfig | None = None,
    lora_config: LoraConfig | None = None,
    remote_config: RemoteConfig | None = None,
) -> str:
    """Run LoRA fine-tuning remotely on a custom HuggingFace Space.

    Creates a Space with the correct dependencies (transformers>=5.0.0 for lfm2
    support, trust_remote_code=True), uploads the training data, and launches
    training on a GPU backend.

    Args:
        model_key: Key in MODEL_REGISTRY (e.g. 'lfm2.5-1.2b-instruct').
        data_path: Path to JSONL training data.
        training_config: Training hyperparameters.
        lora_config: LoRA adapter configuration.
        remote_config: Remote training configuration (token, backend, etc.).

    Returns:
        Hub model ID (e.g. 'username/project-name') where the adapter will be pushed.
    """
    import os
    import tempfile
    from huggingface_hub import HfApi

    from liquid_host.config import get_model_spec

    tc = training_config or TrainingConfig()
    lc = lora_config or LoraConfig()
    rc = remote_config or RemoteConfig()

    spec = get_model_spec(model_key)

    # Resolve HF credentials
    token = rc.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace token required. Set HF_TOKEN env var or pass --hf-token."
        )

    api = HfApi(token=token)
    username = rc.hf_username or api.whoami()["name"]
    dataset_repo = f"{username}/{rc.project_name}-data"
    adapter_repo = f"{username}/{rc.project_name}"
    space_repo = f"{username}/{rc.project_name}-training"

    hw = _HW_MAP.get(rc.backend, rc.backend)

    logger.info("Remote fine-tuning %s (%s)", spec.name, spec.repo_id)
    logger.info("  Space: %s (hardware: %s)", space_repo, hw)
    logger.info("  Dataset: %s", dataset_repo)
    logger.info("  Adapter output: %s", adapter_repo)

    # 1. Upload dataset to HF Hub
    logger.info("Uploading dataset to Hub...")
    dataset = load_chat_dataset(data_path)
    dataset.push_to_hub(dataset_repo, private=rc.private_repo, token=token)
    logger.info("Dataset uploaded: %d examples -> %s", len(dataset), dataset_repo)

    # 2. Generate training script and requirements
    train_script = _build_training_script(
        repo_id=spec.repo_id,
        dataset_repo=dataset_repo,
        adapter_repo=adapter_repo,
        tc=tc,
        lc=lc,
    )

    requirements = (
        "transformers>=5.0.0\n"
        "torch>=2.1.0\n"
        "accelerate>=0.25.0\n"
        "peft>=0.14.0\n"
        "trl>=0.15.0\n"
        "datasets>=3.0.0\n"
        "huggingface-hub>=0.20.0\n"
    )
    if tc.quantize_4bit:
        requirements += "bitsandbytes>=0.45.0\n"

    # 3. Create the Space and upload files
    logger.info("Creating training Space: %s", space_repo)
    api.create_repo(
        space_repo,
        repo_type="space",
        space_sdk="docker",
        space_hardware=hw,
        private=True,
        exist_ok=True,
    )

    # Set HF_TOKEN as a Space secret so the training script can access gated models
    api.add_space_secret(space_repo, "HF_TOKEN", token)

    # Build a minimal Dockerfile that runs the training script
    dockerfile = (
        "FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime\n"
        "WORKDIR /app\n"
        "ENV HF_HOME=/tmp/hf_cache\n"
        "ENV TRANSFORMERS_CACHE=/tmp/hf_cache\n"
        "COPY requirements.txt .\n"
        "RUN pip install --no-cache-dir -r requirements.txt\n"
        "COPY train.py .\n"
        'CMD ["python", "train.py"]\n'
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "train.py").write_text(train_script)
        (tmp_path / "requirements.txt").write_text(requirements)
        (tmp_path / "Dockerfile").write_text(dockerfile)

        api.upload_folder(
            folder_path=tmp,
            repo_id=space_repo,
            repo_type="space",
            commit_message="Launch LoRA fine-tuning for liquid-host",
        )

    logger.info("Training Space created and launched!")
    logger.info("  Monitor: https://huggingface.co/spaces/%s", space_repo)
    logger.info("  Adapter will be pushed to: %s", adapter_repo)

    return adapter_repo


def _detect_target_modules(model) -> list[str]:
    """Auto-detect linear layer names suitable for LoRA."""
    import torch.nn as nn

    target = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get the last part of the name (e.g. "q_proj" from "model.layers.0.self_attn.q_proj")
            short = name.split(".")[-1]
            target.add(short)

    # Remove output/embedding layers that shouldn't get LoRA
    exclude = {"lm_head", "embed_tokens", "wte", "wpe"}
    target -= exclude

    return sorted(target)
