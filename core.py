"""
core.py - Jar "Core" Runtime (2026 Edition)
-------------------------------------------
Features:
- Fixed Similarity Warning: Uses en_core_web_md for true vector support.
- Confidence Fallback: Routes to 'chat' if intent matching is weak.
- Auto-Dependency Repair: Installs missing plugins on-the-fly.
"""

from __future__ import annotations
import importlib.util
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

def _pip_install(packages: Iterable[str]) -> None:
    pkgs = [p for p in packages if p]
    if not pkgs: return
    cmd = [sys.executable, "-m", "pip", "install", "--user", *pkgs]
    subprocess.check_call(cmd)

def _ensure_spacy(model: str = "en_core_web_md") -> Tuple["spacy.language.Language", str]:
    """Ensures spaCy and Medium models (with vectors) are available."""
    try:
        import spacy
    except ImportError:
        _pip_install(["spacy>=3.7.0"])
        import spacy

    try:
        nlp = spacy.load(model)
        return nlp, model
    except Exception:
        print(f"Jarvis: Setting up language model ({model}). Please wait...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
        return spacy.load(model), model

@dataclass(frozen=True)
class Intent:
    name: str
    examples: Tuple[str, ...]
    handler: Callable[[str, "JarCore"], Any]
    plugin: str
    description: str = ""
    threshold: float = 0.55

@dataclass(frozen=True)
class IntentMatch:
    intent: Intent
    score: float
    reason: str

class JarCore:
    def __init__(self, modules_dir: Optional[str] = None, spacy_model: str = "en_core_web_md"):
        self.base_dir = Path(__file__).resolve().parent
        self.modules_dir = Path(modules_dir) if modules_dir else (self.base_dir / "modules")
        self.nlp, self.model_loaded = _ensure_spacy(spacy_model)

        # Shared runtime state (conversation memory, cached data, etc.)
        self.session: Dict[str, Any] = {}
        self.identity: str = "Jarvis, an AI assistant"
        self.session.setdefault("identity", self.identity)

        self._intents: List[Intent] = []
        self._example_docs: Dict[Tuple[str, str], Any] = {}
        self._phrase_matcher = None
        self._plugin_errors: Dict[str, str] = {}

        self.discover_plugins()

    def discover_plugins(self) -> None:
        self._intents.clear()
        self.modules_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.modules_dir.glob("*.py")):
            if path.name.startswith("_"): continue
            self._load_plugin_file(path)
        self._build_matchers()

    def _load_plugin_file(self, path: Path) -> None:
        plugin_name = path.stem
        try:
            spec = importlib.util.spec_from_file_location(f"modules.{plugin_name}", str(path))
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except ModuleNotFoundError as e:
                missing = getattr(e, "name", None)
                if missing:
                    _pip_install([missing])
                    spec.loader.exec_module(module)
                else:
                    raise
            
            intent_dicts = self._extract_intents(module, plugin_name)
            for d in intent_dicts:
                intent = self._intent_from_dict(d, module=module, plugin_name=plugin_name)
                self.register_intent(intent)
        except Exception as e:
            self._plugin_errors[plugin_name] = str(e)

    def _extract_intents(self, module: Any, plugin_name: str) -> List[Dict]:
        found = getattr(module, "INTENTS", [])
        if not found and hasattr(module, "chat_with_jarvis"):
            found = [{"name": "chat", "examples": ["talk", "chat", "hey"], "handler": "chat_with_jarvis"}]
        return found

    def register_intent(self, intent: Intent) -> None:
        self._intents.append(intent)
        for ex in intent.examples:
            self._example_docs[(intent.name, ex)] = self.nlp(ex)

    def _build_matchers(self) -> None:
        from spacy.matcher import PhraseMatcher
        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for intent in self._intents:
            patterns = [self.nlp.make_doc(ex) for ex in intent.examples]
            matcher.add(intent.name, patterns)
        self._phrase_matcher = matcher

    def route(self, user_text: str) -> Optional[IntentMatch]:
        doc = self.nlp(user_text)
        matches = []
        
        phrase_hits = {}
        if self._phrase_matcher is not None:
            phrase_hits = {self.nlp.vocab.strings[m_id]: 1 for m_id, _, _ in self._phrase_matcher(doc)}

        for intent in self._intents:
            p_score = 1.0 if intent.name in phrase_hits else 0.0
            s_score = max([doc.similarity(self._example_docs[(intent.name, ex)]) 
                           for ex in intent.examples if self._example_docs[(intent.name, ex)]] or [0])
            
            final_score = (s_score * 0.6) + (p_score * 0.4)
            matches.append(IntentMatch(intent, final_score, f"sim={s_score:.2f}, phrase={p_score}"))

        return max(matches, key=lambda m: m.score) if matches else None

    def handle(self, user_text: str) -> Any:
        best = self.route(user_text)
        
        # CONFIDENCE FALLBACK: If score < 0.4, force to Chat
        if not best or best.score < 0.4:
            chat = next((i for i in self._intents if i.name == "chat"), None)
            if chat: return chat.handler(user_text, self)
            return "I don't understand."
            
        return best.intent.handler(user_text, self)

    def _intent_from_dict(self, d, module, plugin_name):
        raw = getattr(module, d["handler"]) if isinstance(d["handler"], str) else d["handler"]
        if not callable(raw):
            raise TypeError(f"Handler for intent '{d.get('name')}' is not callable: {raw!r}")

        def _wrapped(user_text: str, core: "JarCore", _fn=raw):
            try:
                return _fn(user_text, core)
            except TypeError:
                try:
                    return _fn(user_text)
                except TypeError:
                    return _fn()

        return Intent(
            d["name"],
            tuple(d["examples"]),
            _wrapped,
            plugin_name,
            threshold=d.get("threshold", 0.55),
        )

if __name__ == "__main__":
    core = JarCore()
    print("Jarvis: Hi - I'm Jarvis, your personal AI assistant. How can I help you today?")
    
    while True:
        cmd = input("You: ").strip()
        if not cmd:
            continue
        if cmd.lower() in ["exit", "quit"]:
            print("Jarvis: Goodbye.")
            break

        try:
            result = core.handle(cmd)
            if result is not None:
                print(f"Jarvis: {str(result)}")
        except Exception:
            print("Jarvis: Something went wrong while handling that request.")