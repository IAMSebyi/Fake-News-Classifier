# Fake News Classifier

This project is a machine learning model that classifies news articles as fake or real. It includes data loading, preprocessing, model training, evaluation, and an API for making predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Usage](#api-usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Fake News Classifier project aims to build a model that can accurately classify news articles as fake or real based on their title and text content. The project includes data preprocessing, feature engineering, model training, evaluation, and a REST API for serving predictions. The base model was trained on the [WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) dataset published in [IEEE Transactions on Computational Social Systems (Volume: 8, Issue: 4, August 2021)](https://ieeexplore.ieee.org/document/9395133).

## Key Features
- Data Loading and Preprocessing: Load and clean the dataset to prepare it for model training.
- Feature Engineering: Extract meaningful features from the text data using techniques like TF-IDF.
- Model Training: Train a machine learning model to classify news articles as fake or real.
- Model Evaluation: Evaluate the model's performance using metrics such as precision, recall, and F1-score.
- API for Predictions: Serve the model using a REST API to classify new news articles.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- git (for cloning the repository)

### Installation

1. **Clone repository**

```bash
git clone https://github.com/IAMSebyi/Fake-News-Classifier.git
cd Fake-News-Classifier
```

2. **Create a virtual environment**
   
```bash
python -m venv venv
source venv/bin/activate    # On Windows use `venv\Scripts\activate`
```

3. **Install dependencies**
   
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Script
To preprocess data, train the model, and evaluate it:

```bash
python main.py
```

## Model Training
If you want to specifically train the model:

```bash
python -m src.models.train
```

## API Usage
To start the API server for making predictions:

```bash
python -m src.app.api
```

Once the server is running, you can make POST requests to http://127.0.0.1:5000/predict with a JSON payload containing the text to classify.

**Example request**:

```json
{
  "text": "Sample news article text to classify."
}
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

### Steps to Contribute

1. Fork the repository
2. Create a new branch (git checkout -b feature-branch)
3. Commit your changes (git commit -m 'Add some feature')
4. Push to the branch (git push origin feature-branch)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
