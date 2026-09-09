from __future__ import annotations
from pathlib import Path
from typing import Tuple
import joblib
import pandas as pd
from .intents import propose_intent, keyword_scores
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

class IntentModel:
    def __init__(self, seed: int=42):
        self.seed = seed
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), min_df=2, max_features=20000, sublinear_tf=True)),
            ('clf', SGDClassifier(loss='log_loss', max_iter=800, tol=1e-3, alpha=1e-5, class_weight='balanced', random_state=seed, n_jobs=-1)),
        ])

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def predict_with_confidence(self, texts):
        probs = self.pipeline.predict_proba(texts)
        idx = probs.argmax(axis=1)
        labels = self.pipeline.classes_[idx]
        conf = probs.max(axis=1)
        # Hybrid override: deterministic domain cues win when they are strong and distinctive.
        for j, text in enumerate(texts):
            scores = keyword_scores(text)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if ranked and ranked[0][1] >= 1.5 and (ranked[0][1] - ranked[1][1]) >= 0.8:
                labels[j] = ranked[0][0]
                conf[j] = max(conf[j], min(0.98, 0.65 + 0.08*ranked[0][1]))
        return labels, conf, probs

    def save(self, path: str | Path):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path):
        obj = cls()
        obj.pipeline = joblib.load(path)
        return obj
