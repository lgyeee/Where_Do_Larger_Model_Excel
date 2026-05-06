# OPENROUTER API Model Map
API_MODEL_MAP = {
    "gemini-3-pro": "google/gemini-3.1-pro-preview",
    "gemini-3-flash": "google/gemini-3-flash",
}
# GROQ API Model Map
GROQ_MODEL_MAP = {
    "gpt-oss-20b": {"model_id": "openai/gpt-oss-20b", "max_tokens": 65536},
    "gpt-oss-120b": {"model_id": "openai/gpt-oss-120b", "max_tokens": 65526},
}


def make_messages(prompt: str, model_id: str, sampling_params: dict) -> dict:
    return {
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
    # n can only be 1 for now
    # openai/gpt-oss-20b and openai/gpt-oss-120b support 'low', 'medium', or 'high'. 'medium' is the default value.
    # How many chat completion choices to generate for each input message. Note that the current moment, only n=1 is supported. Other values will result in a 400 response.
#=============== Set Model Params ===============
def make_sampling_params(max_tokens: int, reasoning_effort: str) -> dict:
    """
    Build sampling parameters.
    """
    kw = {"max_tokens": max_tokens, "reasoning_effort": reasoning_effort}
    kw["temperature"] = 0.7
    kw["top_p"] = 0.95
    if reasoning_effort: 
        kw["include_reasoning"] = True
    else:
        kw["include_reasoning"] = False
    return kw