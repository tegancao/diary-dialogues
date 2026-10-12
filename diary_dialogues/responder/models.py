"""The response models, pinned to exact Hugging Face revisions, plus a fixed-reply baseline.

No response model shares a family with a judge (Mistral, gpt-oss) or with the diarist (Qwen).
"""

RESPONDERS = {
    "llama-3.1-8b": {"model": "meta-llama/Llama-3.1-8B-Instruct",       # gated (Meta license)
                     "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "vllm_kwargs": {}},
    "gemma-3-12b": {"model": "google/gemma-3-12b-it",                     # gated (Gemma terms)
                    "revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
                    "vllm_kwargs": {"limit_mm_per_prompt": {"image": 0}}},   # text only
    "phi-4": {"model": "microsoft/phi-4",                                 # MIT
              "revision": "2db69c1c3e91a05d2c64a3185acfbaf36f744e25", "vllm_kwargs": {}},
    # Larger open models (2026-10-11), same families as above, to separate model size from open vs. closed.
    "gemma-3-27b": {"model": "google/gemma-3-27b-it",                     # gated (Gemma terms)
                    "revision": "005ad3404e59d6023443cb575daa05336842228a",
                    "vllm_kwargs": {"limit_mm_per_prompt": {"image": 0}}, "gpu": "H100"},
    "llama-3.3-70b": {"model": "meta-llama/Llama-3.3-70B-Instruct",       # gated (Meta license)
                      "revision": "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
                      "vllm_kwargs": {"tensor_parallel_size": 4}, "gpu": "H100:4"},
    # Newest open model (2026-10-11): newer training at about Gemma-3-27B's size.
    "gemma-4-31b": {"model": "google/gemma-4-31B-it",                     # Apache-2.0
                    "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
                    # one H100 leaves 7 GB of KV cache, under the 14 GB a 16k-token request needs
                    "vllm_kwargs": {"limit_mm_per_prompt": {"image": 0}, "tensor_parallel_size": 2}, "gpu": "H100:2"},
}

# Frontier reference responders, through their APIs (2026-10-11). Same length cap as SAMPLING, default sampling (no temperature setting), no
# system prompt. Claude shares no family with the judges (Mistral, gpt-oss) or the diarist (Qwen).
API_RESPONDERS = {
    "claude-opus-5.5": {"api": "anthropic", "model": "claude-opus-5-5"},
    "claude-haiku-4.5": {"api": "anthropic", "model": "claude-haiku-4-5-20251001"},
}

# Baseline: the same content-free reply every time, to separate what the replies cause from simply being answered.
BASELINE = "baseline"
BASELINE_REPLY = "Thank you for sharing."

# Same sampling for every model, so differences come from the models, not the settings.
SAMPLING = {"temperature": 0.7, "top_p": 0.95, "max_tokens": 700}
MAX_MODEL_LEN = 16384      # phi-4's limit; 8 entries and 7 replies need about 10K tokens
