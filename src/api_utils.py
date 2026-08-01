import inspect

# OPENROUTER API Model Map
OPENROUTER_MODEL_MAP = {
    "gemini-3-pro": {"model_id": "google/gemini-3.1-pro-preview", "max_tokens": 65536},
    "gemini-3-flash": {"model_id": "google/gemini-3-flash-preview", "max_tokens": 65536},
    "gpt-oss-20b": {"model_id": "openai/gpt-oss-20b", "max_tokens": 65536},
    "gpt-oss-120b": {"model_id": "openai/gpt-oss-120b", "max_tokens": 65526},
    "gpt-oss-20b-free": {"model_id": "openai/gpt-oss-20b:free", "max_tokens": 65536},
    "deepseek-v4-pro": {"model_id": "deepseek/deepseek-v4-pro", "max_tokens": 65536},
}
# GROQ API Model Map
GROQ_MODEL_MAP = {
    "gpt-oss-20b": {"model_id": "openai/gpt-oss-20b", "max_tokens": 65536},
    "gpt-oss-120b": {"model_id": "openai/gpt-oss-120b", "max_tokens": 65526},
    "qwen3-32b": {"model_id": "qwen/qwen3-32b", "max_tokens": 40960},
}


def _called_from_batch_generator() -> bool:
    """True when make_groq_messages is building a Groq Batch API jsonl body."""
    for frame in inspect.stack()[1:]:
        if frame.function == "make_groq_messages":
            continue
        if frame.filename.endswith("gen_batch_requests.py"):
            return True
    return False


def make_groq_messages(prompt: str, model_id: str, sampling_params: dict) -> dict:
    is_batch = _called_from_batch_generator()
    body = {
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "model": model_id,
        "n": 1,
        "max_completion_tokens": sampling_params["max_tokens"] - 4096,
        "reasoning_effort": sampling_params["reasoning_effort"],
        "include_reasoning": sampling_params["include_reasoning"],
        "temperature": sampling_params["temperature"],
        "top_p": sampling_params["top_p"],
    }
    # Groq Batch API uses its own batch tier; service_tier is for sync chat only.
    if not is_batch:
        body["service_tier"] = "on_demand"
    return body


def make_openrouter_messages(prompt: str, model_id: str, sampling_params: dict) -> dict:
    body = {
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "model": model_id,
        "temperature": sampling_params["temperature"],
        "top_p": sampling_params["top_p"],
        "max_tokens": sampling_params["max_tokens"] - 4096,
    }
    if sampling_params.get("reasoning_effort"):
        body["extra_body"] = {"reasoning": {"effort": sampling_params["reasoning_effort"]}}
    return body


#=============== Set Model Params ===============
def make_sampling_params(max_tokens: int, reasoning_effort: str, model_id: str) -> dict:
    """
    Build sampling parameters.
    """
    if model_id == "qwen3-32b":
        kw = {"max_tokens": max_tokens, "reasoning_effort": reasoning_effort}
        kw["temperature"] = 0.6
        kw["top_p"] = 0.95
    elif model_id in ("gpt-oss-20b", "gpt-oss-120b", "gpt-oss-20b-free"):
        kw = {"max_tokens": max_tokens, "reasoning_effort": reasoning_effort}
        kw["temperature"] = 0.7
        kw["top_p"] = 0.95
    elif model_id in ("gemini-3-pro", "gemini-3-flash"):
        kw = {"max_tokens": max_tokens, "reasoning_effort": reasoning_effort}
        kw["temperature"] = 0.7
        kw["top_p"] = 0.95
    elif model_id in ("deepseek-v4-pro",):
        kw = {"max_tokens": max_tokens, "reasoning_effort": reasoning_effort}
        kw["temperature"] = 0.7
        kw["top_p"] = 0.95
    else:
        raise ValueError(f"Unknown model for sampling params: {model_id}")

    if reasoning_effort:
        kw["include_reasoning"] = True
    else:
        kw["include_reasoning"] = False
    return kw
