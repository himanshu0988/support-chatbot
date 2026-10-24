"""scikit-learn intent classifier (TF-IDF + Logistic Regression)."""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .knowledge_base import KnowledgeBase
from .preprocessing import preprocess

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "intent_model.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", FeatureUnion([  # word n-grams + character n-grams (robust to typos)
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(max_iter=1000, C=10.0)),
    ])


class IntentClassifier:
    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline

    def train(self, patterns: list[str], labels: list[str]) -> "IntentClassifier":
        self.pipeline = build_pipeline().fit([preprocess(p) for p in patterns], labels)
        return self

    def predict(self, query: str) -> tuple[str, float]:
        """Return (intent, confidence)."""
        proba = self.pipeline.predict_proba([preprocess(query)])[0]
        idx = int(proba.argmax())
        return self.pipeline.classes_[idx], float(proba[idx])

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load_or_train(cls, kb: KnowledgeBase, path: Path = MODEL_PATH) -> "IntentClassifier":
        if path.exists():
            return cls(joblib.load(path))
        clf = cls().train(*kb.training_data())
        clf.save(path)
        return clf
