import joblib


class ModelService:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text: str) -> int:
        features = self.vectorizer.transform([text])
        prediction = self.model.predict(features)[0]
        return prediction
