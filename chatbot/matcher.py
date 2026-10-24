"""TF-IDF + cosine similarity matcher: finds the closest known pattern."""
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .knowledge_base import KnowledgeBase
from .preprocessing import preprocess


@dataclass
class Match:
    tag: str
    score: float
    pattern: str


class SimilarityMatcher:
    def __init__(self, kb: KnowledgeBase):
        patterns, labels = kb.training_data()
        self.patterns, self.labels = patterns, labels
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([preprocess(p) for p in patterns])

    def match(self, query: str) -> Match:
        vec = self.vectorizer.transform([preprocess(query)])
        scores = cosine_similarity(vec, self.matrix).ravel()
        best = int(scores.argmax())
        return Match(self.labels[best], float(scores[best]), self.patterns[best])
