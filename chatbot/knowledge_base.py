"""Loads the JSON knowledge base (intents, patterns, responses)."""
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "intents.json"


class KnowledgeBase:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.intents = {i["tag"]: i for i in data["intents"]}

    def training_data(self) -> tuple[list[str], list[str]]:
        """Return (patterns, labels) for training."""
        X, y = [], []
        for tag, intent in self.intents.items():
            for pattern in intent["patterns"]:
                X.append(pattern)
                y.append(tag)
        return X, y

    def responses(self, tag: str) -> list[str]:
        return self.intents[tag]["responses"]
