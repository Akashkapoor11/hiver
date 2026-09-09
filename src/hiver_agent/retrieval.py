from __future__ import annotations
from pathlib import Path
import joblib, numpy as np, pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class HistoricalRetriever:
    def __init__(self, max_docs=30000, seed=42):
        self.max_docs = max_docs
        self.seed = seed
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), min_df=2, max_features=20000, sublinear_tf=True)
        self.index = None
        self.cases = None
        self.excluded_ids = set()

    def fit(self, cases: pd.DataFrame):
        cases = cases.copy()
        cases = cases[cases['customer_text'].fillna('').str.len() >= 8]
        cases['evidence_quality'] = cases['support_text'].fillna('').map(self._quality)
        if len(cases) > self.max_docs:
            cases = cases.sample(self.max_docs, random_state=self.seed)
        self.cases = cases.reset_index(drop=True)
        X = self.vectorizer.fit_transform(self.cases['customer_text'].fillna(''))
        self.index = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=min(20, len(self.cases)))
        self.index.fit(X)
        return self

    def set_excluded_ids(self, ids):
        self.excluded_ids = {str(x) for x in ids}
        return self

    def search(self, text: str, k: int=5):
        if self.index is None or self.cases is None or len(self.cases)==0:
            return []
        q = self.vectorizer.transform([text or ''])
        dist, idx = self.index.kneighbors(q, n_neighbors=min(k, len(self.cases)))
        out = []
        candidates=[]
        for d, i in zip(dist[0], idx[0]):
            row = self.cases.iloc[int(i)].to_dict()
            if str(row.get('customer_tweet_id','')) in self.excluded_ids:
                continue
            row['similarity'] = float(1-d)
            candidates.append(row)
        # Favor substantive historical resolutions over one-line routing replies.
        candidates.sort(key=lambda r: 0.8*r['similarity'] + 0.2*r.get('evidence_quality',0.0), reverse=True)
        for row in candidates[:k]:
            row['similarity'] = round(row['similarity'], 4)
            out.append(row)
        return out

    @staticmethod
    def _quality(text: str) -> float:
        t = str(text or '')
        if not t: return 0.0
        has_action = bool(re.search(r'\b(try|check|go to|open|restart|update|reset|disable|enable|send|contact|select|tap|follow)\b', t, re.I))
        generic = bool(re.search(r'(?i)^(thanks|we can help|please dm|send us a dm|reach out)', t.strip())) and len(t) < 120
        score = min(1.0, len(t)/240.0) + (0.35 if has_action else 0) - (0.35 if generic else 0)
        return max(0.0, min(1.0, score))

    def save(self, path: str | Path):
        joblib.dump({'vectorizer': self.vectorizer, 'index': self.index, 'cases': self.cases}, path)

    @classmethod
    def load(cls, path: str | Path):
        payload = joblib.load(path)
        obj = cls()
        obj.vectorizer = payload['vectorizer']; obj.index = payload['index']; obj.cases = payload['cases']
        return obj
