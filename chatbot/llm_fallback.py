"""Optional LLM fallback for the hybrid architecture.

Used only when rule-based confidence is low. Needs `pip install anthropic`
and the ANTHROPIC_API_KEY environment variable. Without them it returns None
and the dialogue manager uses a polite canned fallback.
"""
import os

SYSTEM_PROMPT = (
    "You are a customer support assistant for an online store called ShopEasy. "
    "Answer briefly and politely. If you do not know company-specific details "
    "(order status, account data), do not invent them; offer to connect the user "
    "to a human agent instead."
)


class LLMFallback:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("LLM_MODEL", "claude-sonnet-5-5")
        self.client = None
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.client = anthropic.Anthropic()
            except ImportError:
                pass

    @property
    def available(self) -> bool:
        return self.client is not None

    def ask(self, query: str, history: list[dict] | None = None) -> str | None:
        if not self.available:
            return None
        messages = (history or [])[-6:] + [{"role": "user", "content": query}]
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=300,
                system=SYSTEM_PROMPT, messages=messages,
            )
            return resp.content[0].text
        except Exception:  # network/API errors must never crash the bot
            return None
