from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    data_transformed = vectorizer.transform([data])
    prediction = model.predict(data_transformed)
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
