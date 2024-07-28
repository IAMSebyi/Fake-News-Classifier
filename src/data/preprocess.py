import string

import pandas as pd


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def clean_text(text: str) -> str:
        # Implement text cleaning steps
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        return text

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Handle missing values: fill NaNs with empty strings
        df['title'].fillna('', inplace=True)
        df['text'].fillna('', inplace=True)

        # Concatenate 'title' and 'text' columns
        df['combined_text'] = df['title'] + ' ' + df['text']

        # Clean the concatenated text
        df['cleaned_text'] = df['combined_text'].apply(self.clean_text)

        return df
