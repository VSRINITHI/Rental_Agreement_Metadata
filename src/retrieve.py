"""Select the most similar labeled training documents to use as few-shot examples."""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Example:
    stem: str
    text: str
    labels: dict[str, str]


class ExampleRetriever:
    """TF-IDF nearest-neighbour search over the training documents."""

    def __init__(self, examples: list[Example]):
        self.examples = examples
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        if examples:
            self._vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1)
            self._matrix = self._vectorizer.fit_transform([e.text for e in examples])

    def top_k(self, text: str, k: int, exclude_stem: str | None = None) -> list[Example]:
        """k most similar examples. An example with the same stem as the target
        is always skipped, which prevents label leakage during evaluation."""
        if not self.examples or k <= 0 or self._vectorizer is None:
            return []
        scores = cosine_similarity(self._vectorizer.transform([text]), self._matrix)[0]
        order = sorted(range(len(self.examples)), key=lambda i: scores[i], reverse=True)
        picked = [self.examples[i] for i in order if self.examples[i].stem != exclude_stem]
        return picked[:k]
