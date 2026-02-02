import os
import subprocess
import sys
from pathlib import Path
import re
from typing import Optional

def _pip_install(packages):
    cmd = [sys.executable, "-m", "pip", "install", "--user", *packages]
    subprocess.check_call(cmd)

try:
    import requests
except ImportError:
    _pip_install(["requests"])
    import requests

# API Key configuration
API_KEY = "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"
url = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "qwen/qwen3-32b"

SYSTEM_PROMPT = (
    "You are Jarvis, a friendly, helpful, and highly capable personal AI assistant. "
    "Always respond politely and clearly, with concise and actionable answers. "
    "Never reveal internal reasoning. If you need to think, put it ONLY inside <think>...</think> tags. "
    "Your output to the user must contain ONLY the final answer (no analysis, no scratchpad, no meta commentary)."
)

def _read_dotenv_key() -> Optional[str]:
    try:
        root = Path(__file__).resolve().parents[1]
        env_path = root / ".env"
        if not env_path.exists():
            return None
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "GROQ_API_KEY":
                val = v.strip().strip('"').strip("'")
                return val or None
        return None
    except Exception:
        return None

def _get_api_key() -> Optional[str]:
    direct = (API_KEY or "").strip()
    return os.environ.get("GROQ_API_KEY") or _read_dotenv_key() or (direct or None)

def _clean_model_output(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    # Remove <think> blocks
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE).strip()
    # Handle final answer markers
    for marker in ("FINAL ANSWER:", "Final Answer:", "FINAL:", "Final:"):
        if marker in s:
            s = s.split(marker)[-1].strip()
    return s

def handle_chat(user_text: str, core) -> str:
    api_key = _get_api_key()
    if not api_key:
        return "To enable chat, set `GROQ_API_KEY`."

    # Initialize or retrieve conversation history from the session
    conversation = core.session.setdefault("conversation", [{"role": "system", "content": SYSTEM_PROMPT}])
    
    # Add the current user message
    conversation.append({"role": "user", "content": user_text})

    payload = {
        "model": MODEL,
        "messages": conversation,
        "max_tokens": 500,
        "temperature": 0.8,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        ai_message = data["choices"][0]["message"]["content"]
        ai_message = _clean_model_output(ai_message)
        
        # Save response to history
        conversation.append({"role": "assistant", "content": ai_message})
        return ai_message
    except Exception as e:
        return f"Chat failed: {str(e)}"

INTENTS = [
    {
        "name": "chat",
        "examples": ["chat", "talk to jarvis", "let's talk", "hey", "hello", "hi", "help"],
        "handler": "handle_chat",
        "threshold": 0.2,
        "description": "General conversation with Jarvis.",
    }
]

if __name__ == "__main__":
    class _Core:
        session = {}
    core = _Core()
    while True:
        t = input("You: ").strip()
        if t.lower() in {"exit", "quit"}:
            break
        print("Jarvis:", handle_chat(t, core))