# Sentiment Analysis of IMDb Movie Reviews Using Apache Spark and PySpark Machine Learning

This repository contains a complete DASC6021 Big Data Analytics & Data Visualization group project submission.

## Submission Contents

- `notebooks/Sentiment_Analysis_IMDb_PySpark.ipynb` - well-documented Jupyter Notebook with Spark loading, preprocessing, TF-IDF, model training, evaluation, and visualizations.
- `reports/Report_IMDb_Sentiment_PySpark.md` - 8-10 page written report draft in academic format.
- `slides/index.html` - browser-based online presentation slides.
- `data/README.md` - dataset placement instructions.
- `requirements.txt` - Python dependencies.

## Dataset

Place the IMDb Movie Reviews CSV file at:

```text
data/imdb_reviews.csv
```

The notebook expects a text column such as `review`, `text`, `content`, or `comment`, and a sentiment column such as `sentiment`, `label`, `class`, or `target`. Labels may be `positive`/`negative` or `1`/`0`.

## Environment Setup

1. Install Java JDK 8, 11, or 17.
2. Install Apache Spark or use PySpark from pip.
3. Create and activate a Python environment.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Start Jupyter:

```bash
jupyter notebook
```

6. Open:

```text
notebooks/Sentiment_Analysis_IMDb_PySpark.ipynb
```

7. Run all cells from top to bottom.

## Expected Workflow

The notebook performs the following tasks:

1. Starts a Spark session.
2. Loads `imdb_reviews.csv` into a Spark DataFrame.
3. Cleans review text and standardizes sentiment labels.
4. Tokenizes text and removes stopwords.
5. Converts text into TF-IDF features.
6. Trains Logistic Regression, Naive Bayes, and Decision Tree models using PySpark ML.
7. Evaluates models using Accuracy, Precision, Recall, and F1-score.
8. Visualizes class balance, review length distribution, model comparison, and confusion matrix.
9. Summarizes findings and model limitations.

## Notes for Submission

- Replace placeholder group member names in the report and slides before submitting.
- Run the notebook on the final dataset so charts and metric tables contain executed outputs.
- Export the report to PDF or DOCX if required by the instructor.
- Open `slides/index.html` in a browser for presentation mode.
