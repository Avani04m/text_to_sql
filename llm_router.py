"""
Thin router over Groq / OpenAI / Gemini.

Every call goes through `call_structured()`, which forces the model to
return JSON and validates it into a Pydantic model before anything else
in the pipeline touches it. If parsing/validation fails, we retry with
the validation error fed back to the model. Transient network errors
(timeouts, connection drops — common on flaky/corporate networks) are
retried separately with a short backoff, since those have nothing to do
with the model's output and don't need a reworded prompt.
"""
from __future__ import annotations
import json
import re
import time
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# Suggested default routing (see README for rationale):
#   ambiguity detection -> groq   (fast, cheap, simple classification)
#   sql generation      -> openai or gemini (stronger multi-step reasoning)
#   answer formatting    -> groq   (fast, low-stakes)
# Note: Groq's model lineup changes over time and older model IDs get retired
# or moved to Enterprise-only pricing (e.g. llama-3.3-70b-versatile). If you
# get a 404 "model_not_found" error, check https://console.groq.com/docs/models
# for the current list and update these IDs.
DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
}

REQUEST_TIMEOUT_SECONDS = 45
NETWORK_RETRY_ATTEMPTS = 3
NETWORK_RETRY_BACKOFF_SECONDS = 2


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _call_groq(system: str, user: str, api_key: str, model: str) -> str:
    from groq import Groq
    client = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return resp.choices[0].message.content


def _call_openai(system: str, user: str, api_key: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return resp.choices[0].message.content


def _call_gemini(system: str, user: str, api_key: str, model: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model_name=model, system_instruction=system)
    resp = gmodel.generate_content(
        user,
        generation_config={"response_mime_type": "application/json", "temperature": 0},
        request_options={"timeout": REQUEST_TIMEOUT_SECONDS},
    )
    return resp.text


_PROVIDER_FNS = {"groq": _call_groq, "openai": _call_openai, "gemini": _call_gemini}


def _is_network_error(e: Exception) -> bool:
    """True for connection/timeout-type errors worth a plain retry (no prompt change)."""
    name = type(e).__name__
    return any(
        keyword in name
        for keyword in ("Connection", "Timeout", "APIConnectionError", "APITimeoutError")
    )


def _call_with_network_retries(fn, full_system: str, user_prompt: str, api_key: str, model: str) -> str:
    last_error = None
    for attempt in range(NETWORK_RETRY_ATTEMPTS):
        try:
            return fn(full_system, user_prompt, api_key, model)
        except Exception as e:
            if not _is_network_error(e):
                raise  # not a network issue — let call_structured's JSON-retry logic handle it, or bubble up
            last_error = e
            if attempt < NETWORK_RETRY_ATTEMPTS - 1:
                time.sleep(NETWORK_RETRY_BACKOFF_SECONDS * (attempt + 1))
    # Surface the real underlying cause, not just the SDK's generic "Connection error."
    cause = getattr(last_error, "__cause__", None)
    detail = f" (underlying: {type(cause).__name__}: {cause})" if cause else ""
    raise RuntimeError(
        f"Could not reach the API after {NETWORK_RETRY_ATTEMPTS} attempts: "
        f"{type(last_error).__name__}: {last_error}{detail}. "
        f"This is usually a network/firewall/proxy issue rather than a bug in the app — "
        f"try test_connection.py to isolate it further."
    )


def call_structured(
    provider: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    model: str | None = None,
) -> T:
    """Call `provider` and parse+validate the response into `schema`.
    Network errors get their own short retry loop (see above); once a response
    actually comes back, invalid JSON/schema gets one retry with the error fed
    back into the prompt."""
    if provider not in _PROVIDER_FNS:
        raise ValueError(f"Unknown provider: {provider}")
    if not api_key:
        raise RuntimeError(f"No API key set for '{provider}'. Add one in the sidebar.")
    model = model or DEFAULT_MODELS[provider]
    fn = _PROVIDER_FNS[provider]

    schema_hint = (
        f"\n\nRespond ONLY with a single JSON object matching this shape "
        f"(no markdown, no commentary):\n{json.dumps(schema.model_json_schema(), indent=2)}"
    )
    full_system = system_prompt + schema_hint

    last_error = None
    for attempt in range(2):
        raw = _call_with_network_retries(fn, full_system, user_prompt, api_key, model)
        try:
            cleaned = _strip_json_fences(raw)
            data = json.loads(cleaned)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            user_prompt = (
                user_prompt
                + f"\n\nYour previous response was invalid ({e}). "
                  f"Return ONLY valid JSON matching the required schema."
            )
    raise RuntimeError(f"LLM did not return valid structured output after retries: {last_error}")