"""
LLM client abstraction.

Supports four modes:
    1. Google Gemini     (set GEMINI_API_KEY)  <- generous free tier
    2. Anthropic Claude  (set ANTHROPIC_API_KEY)
    3. OpenAI GPT        (set OPENAI_API_KEY)
    4. Rule-based        (no key needed — deterministic fallback logic)

Rule-based mode is important: it lets the whole agentic pipeline run and be
evaluated without any API cost, and it also serves as an ablation
baseline showing what the LLM reasoning actually adds.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Optional, List


class LLMClient:
    """
    Thin wrapper over whichever LLM provider is configured.

    All agents call `complete_json()` which guarantees a dict comes back,
    falling back to a default structure if parsing fails.
    """

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        # Load .env if present
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
            except ImportError:
                pass

        self.provider = provider or os.getenv("LLM_PROVIDER", "auto")
        self.model = model or os.getenv("LLM_MODEL")
        self._client = None
        self.call_count = 0

        # Auto-detect provider from available keys
        if self.provider == "auto":
            if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                self.provider = "gemini"
            elif os.getenv("ANTHROPIC_API_KEY"):
                self.provider = "anthropic"
            elif os.getenv("OPENAI_API_KEY"):
                self.provider = "openai"
            else:
                self.provider = "rules"

        self._init_client()

    def _init_client(self):
        if self.provider == "gemini":
            try:
                from google import genai
                key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                self._client = genai.Client(api_key=key)
                self.model = self.model or "gemini-3.6-flash"
                self._genai_sdk = "new"
                print(f"🤖 LLM: Google Gemini ({self.model})")
            except ImportError:
                # Fall back to the older google-generativeai SDK
                try:
                    import google.generativeai as genai_old
                    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                    genai_old.configure(api_key=key)
                    self.model = self.model or "gemini-3.6-flash"
                    self._client = genai_old.GenerativeModel(self.model)
                    self._genai_sdk = "old"
                    print(f"🤖 LLM: Google Gemini ({self.model}, legacy SDK)")
                except Exception as e:
                    print(f"⚠️  Gemini init failed ({e}). Falling back to rule-based.")
                    self.provider = "rules"
            except Exception as e:
                print(f"⚠️  Gemini init failed ({e}). Falling back to rule-based.")
                self.provider = "rules"

        elif self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                self.model = self.model or "claude-sonnet-4-6"
                print(f"🤖 LLM: Anthropic ({self.model})")
            except Exception as e:
                print(f"⚠️  Anthropic init failed ({e}). Falling back to rule-based.")
                self.provider = "rules"

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.model = self.model or "gpt-4o-mini"
                print(f"🤖 LLM: OpenAI ({self.model})")
            except Exception as e:
                print(f"⚠️  OpenAI init failed ({e}). Falling back to rule-based.")
                self.provider = "rules"

        if self.provider == "rules":
            print("🤖 LLM: rule-based fallback (no API key set)")

    @property
    def is_llm(self) -> bool:
        """True if an actual LLM is in use (not the rule-based fallback)."""
        return self.provider in ("gemini", "anthropic", "openai")

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback: Dict,
        max_tokens: int = 800,
    ) -> Dict:
        """
        Get a JSON response from the LLM.

        Args:
            system_prompt: Role/instructions for the model
            user_prompt: The actual query with context
            fallback: Dict returned if LLM unavailable or parsing fails
            max_tokens: Response length cap

        Returns:
            Parsed dict — never raises.
        """
        if not self.is_llm:
            return fallback

        try:
            self.call_count += 1
            raw = self._call(system_prompt, user_prompt, max_tokens)
            parsed = self._extract_json(raw)
            return parsed if parsed else fallback
        except Exception as e:
            print(f"⚠️  LLM call failed: {e}")
            return fallback

    def _call(self, system: str, user: str, max_tokens: int) -> str:
        if self.provider == "gemini":
            # Gemini has no separate system role, so prepend it to the prompt
            combined = f"{system}\n\n---\n\n{user}"
            if getattr(self, "_genai_sdk", "new") == "new":
                from google.genai import types as gtypes
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=combined,
                    config=gtypes.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.3,
                        response_mime_type="application/json",
                    ),
                )
                return resp.text or ""
            else:
                resp = self._client.generate_content(
                    combined,
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": 0.3,
                        "response_mime_type": "application/json",
                    },
                )
                return resp.text or ""

        elif self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if hasattr(b, "text"))

        elif self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content

        return ""

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """Pull a JSON object out of the response, tolerating markdown fences."""
        if not text:
            return None

        # Strip markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None