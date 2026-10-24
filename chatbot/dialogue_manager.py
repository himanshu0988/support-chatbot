"""Dialogue management: rules -> classifier + similarity -> LLM/canned fallback."""
import random
import re
from dataclasses import dataclass, field

from .intent_classifier import IntentClassifier
from .knowledge_base import KnowledgeBase
from .llm_fallback import LLMFallback
from .matcher import SimilarityMatcher

ORDER_ID_RE = re.compile(r"#?\b(?:ORD-?)?(\d{5,10})\b", re.IGNORECASE)
HUMAN_RE = re.compile(r"\b(human|agent|representative|real person|live chat)\b", re.IGNORECASE)

CLASSIFIER_THRESHOLD = 0.35   # min classifier probability
SIMILARITY_THRESHOLD = 0.30   # min cosine similarity to nearest pattern

FALLBACK_REPLIES = [
    "Sorry, I didn't quite get that. Could you rephrase? I can help with orders, returns, shipping, payments and your account.",
    "I'm not sure I understood. I can help with order tracking, refunds, delivery and payments. What do you need?",
]


@dataclass
class Reply:
    text: str
    intent: str | None
    confidence: float
    source: str  # "rule" | "model" | "llm" | "fallback"


@dataclass
class DialogueManager:
    kb: KnowledgeBase = field(default_factory=KnowledgeBase)
    use_llm: bool = True

    def __post_init__(self):
        self.classifier = IntentClassifier.load_or_train(self.kb)
        self.matcher = SimilarityMatcher(self.kb)
        self.llm = LLMFallback() if self.use_llm else None
        self.order_id: str | None = None   # simple slot memory
        self.last_intent: str | None = None
        self.history: list[dict] = []

    def _respond_intent(self, tag: str) -> str:
        text = random.choice(self.kb.responses(tag))
        if self.order_id:
            text = text.replace("{order_id}", f"#{self.order_id}")
        else:
            text = text.replace("{order_id}", "your order")
        return text

    def reply(self, message: str) -> Reply:
        message = message.strip()
        if not message:
            return Reply("Please type a message so I can help.", None, 0.0, "rule")

        # Rule 1: capture an order id (slot filling)
        m = ORDER_ID_RE.search(message)
        if m:
            self.order_id = m.group(1)

        # Rule 2: explicit request for a human
        if HUMAN_RE.search(message):
            return self._finish(message, "human_agent", 1.0, "rule")

        # Rule 3: message is only an order id -> continue previous topic
        if m and len(message.split()) <= 2 and self.last_intent in {"order_status", "cancel_order", "return_refund"}:
            return self._finish(message, self.last_intent, 1.0, "rule")

        # Statistical layer: classifier + TF-IDF cosine similarity
        tag, conf = self.classifier.predict(message)
        match = self.matcher.match(message)
        agree = match.tag == tag
        if (conf >= CLASSIFIER_THRESHOLD and match.score >= SIMILARITY_THRESHOLD) and (
            agree or conf >= 0.6
        ):
            return self._finish(message, tag, conf, "model")
        if match.score >= 0.6:  # near-exact pattern hit overrides weak classifier
            return self._finish(message, match.tag, match.score, "model")

        # Hybrid fallback: LLM if configured, otherwise canned reply
        if self.llm and self.llm.available:
            answer = self.llm.ask(message, self.history)
            if answer:
                return self._finish(message, None, conf, "llm", text=answer)
        return self._finish(message, None, conf, "fallback", text=random.choice(FALLBACK_REPLIES))

    def _finish(self, message, tag, conf, source, text=None) -> Reply:
        if text is None:
            text = self._respond_intent(tag)
            self.last_intent = tag
        self.history += [{"role": "user", "content": message},
                         {"role": "assistant", "content": text}]
        return Reply(text, tag, conf, source)
