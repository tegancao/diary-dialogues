"""Shared Modal set-up, so every experiment runs in the same environment.

Pinned versions keep GPU runs reproducible. Model weights are cached in a Modal volume, so they download only once.
"""
import modal

from diary_dialogues.paths import VECTOR_DIR

HF_CACHE = modal.Volume.from_name("diary-dialogues-hf-cache", create_if_missing=True)
# Long jobs also save their results here, so a dropped local connection (run with `modal run --detach`) loses nothing:
#   modal volume get diary-dialogues-results <file> .
RESULTS = modal.Volume.from_name("diary-dialogues-results", create_if_missing=True)
VOLUMES = {"/hf": HF_CACHE, "/results": RESULTS}

# transformers image: for the steered diarist (activation steering needs direct access to the model's layers)
transformers_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "transformers==5.19.0", "accelerate")
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("diary_dialogues")
)
# For calls to a responder API (run.reply_anthropic): the same base, since the harness imports torch; no GPU needed.
api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "transformers==5.19.0", "accelerate", "anthropic==1.13.0")
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("diary_dialogues")
)


def with_vectors(image: modal.Image) -> modal.Image:
    """Add the schema vectors (run scripts/download_data.py first)."""
    return image.add_local_dir(VECTOR_DIR, "/vectors")

# vLLM image: for judges and response models (fast batched inference; no steering needed).
# Built on NVIDIA's CUDA "devel" image, which includes the CUDA compiler (nvcc): vLLM/FlashInfer compile some GPU kernels
# on first use (sampling, CUDA graphs, multi-GPU all-reduce) and fail on a slim image without it.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.31.0", "huggingface_hub")
    .env({"HF_HOME": "/hf", "VLLM_LOGGING_LEVEL": "WARNING", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("diary_dialogues")
)

# Judges, pinned to exact Hugging Face revisions. vllm_kwargs follow each model card's recommended vLLM settings.
JUDGES = {
    "mistral-small-3.1": {
        "model": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "revision": "68faf511d618ef198fef186659617cfd2eb8e33a",
        "vllm_kwargs": {"tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral"},
        "gpu": "H100",
    },
    "gpt-oss-120b": {                                   # OpenAI open-weight reasoning model, Apache-2.0
        "model": "openai/gpt-oss-120b",
        "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
        "vllm_kwargs": {},                              # MXFP4 weights (~61 GB) fit one 80 GB GPU
        "reasoning": {"chat_kwargs": {"reasoning_effort": "medium"},
                      "answer_marker": "<|channel|>final<|message|>", "max_tokens": 2048},
        "gpu": "H100",
    },
    # Pilot-only (2026-10-10): Llama-3.1-8B is a response model, so a Llama judge would grade its own family.
    "llama-3.3-70b": {                                  # gated: needs an approved Hugging Face account
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "revision": "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        # 140 GB of bf16 weights across 2 x 80 GB GPUs leave little room for the KV cache: use 95% of GPU memory
        "vllm_kwargs": {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.95},
        "gpu": "H100:2",
    },
}


def hf_token_secret() -> modal.Secret:
    """Pass the local Hugging Face login to the container, for gated models (e.g. Llama).

    The token is read from your local login (`hf auth login`) when you launch a run; it is never printed or stored in
    the repo. Inside the container this evaluates to an empty secret, which keeps Modal's dependency lists identical
    on both sides.
    """
    token = None
    if modal.is_local():
        from huggingface_hub import get_token
        token = get_token()
    return modal.Secret.from_dict({"HF_TOKEN": token} if token else {})
