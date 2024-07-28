from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class FeatureEngineer:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def save_vectorizer(self, filepath):
        joblib.dump(self.vectorizer, filepath)

    def load_vectorizer(self, filepath):
        self.vectorizer = joblib.load(filepath)
