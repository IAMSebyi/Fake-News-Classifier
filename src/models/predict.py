import joblib
import pandas as pd


class Predictor:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, texts: pd.Series) -> pd.Series:
        features = self.vectorizer.transform(texts)
        predictions = self.model.predict(features)
        return predictions
