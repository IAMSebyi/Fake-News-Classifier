import warnings

from sklearn.model_selection import train_test_split

from src.data.feature_engineer import FeatureEngineer
from src.data.load_data import DataLoader
from src.data.preprocess import Preprocessor
from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer
from src.utils.utils import save_data, load_data

from os.path import exists

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    # Define dataset paths
    raw_path = 'data/raw/dataset.csv'
    processed_path = 'data/processed/preprocessed_dataset.csv'

    # Check if processed dataset exists
    if exists(processed_path):
        # Load preprocessed dataset
        df = load_data(processed_path)
    else:
        # Load raw dataset
        loader = DataLoader(raw_path)
        df = loader.load_data()

        # Preprocess data
        preprocessor = Preprocessor()
        df = preprocessor.preprocess_data(df)

        # Save preprocessed dataset
        save_data(df, 'data/processed/preprocessed_dataset.csv')

    # Feature engineering
    engineer = FeatureEngineer()
    X = engineer.fit_transform(df['cleaned_text'])
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    trainer.save_model('models/fake_news_model.pkl')
    engineer.save_vectorizer('models/tfidf_vectorizer.pkl')

    # Evaluate model
    evaluator = ModelEvaluator(trainer.model)
    evaluator.evaluate(X_test, y_test)


if __name__ == '__main__':
    main()
