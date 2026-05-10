import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "imdb_sentiment_pyspark_project"


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip(), encoding="utf-8")


def notebook_cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": dedent(source).lstrip().splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def build_notebook():
    cells = [
        notebook_cell(
            "markdown",
            r"""
            # Sentiment Analysis of IMDb Movie Reviews Using Apache Spark and PySpark Machine Learning

            **Course:** DASC6021 Big Data Analytics & Data Visualization  
            **Tools:** Apache Spark, PySpark MLlib, Jupyter Notebook  
            **Dataset:** `data/imdb_reviews.csv`

            This notebook implements a scalable sentiment analysis pipeline for IMDb movie reviews. It loads a CSV dataset into a Spark DataFrame, cleans and tokenizes text, removes stopwords, extracts TF-IDF features, trains Logistic Regression, Naive Bayes, and Decision Tree classifiers, and evaluates them using accuracy, precision, recall, and F1-score.
            """,
        ),
        notebook_cell(
            "markdown",
            r"""
            ## 1. Environment Setup

            The notebook assumes that Java, Apache Spark, PySpark, and Jupyter are installed. The accompanying `README.md` explains installation steps. Run all cells from top to bottom after placing `imdb_reviews.csv` in the `data/` folder.
            """,
        ),
        notebook_cell(
            "code",
            r"""
            import os
            import re
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd

            from pyspark.sql import SparkSession
            from pyspark.sql import functions as F
            from pyspark.sql.types import DoubleType

            from pyspark.ml import Pipeline
            from pyspark.ml.classification import LogisticRegression, NaiveBayes, DecisionTreeClassifier
            from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
            from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF

            spark = (
                SparkSession.builder
                .appName("IMDb Sentiment Analysis with PySpark")
                .config("spark.sql.shuffle.partitions", "8")
                .getOrCreate()
            )

            spark.sparkContext.setLogLevel("WARN")
            print("Spark version:", spark.version)
            """,
        ),
        notebook_cell(
            "markdown",
            r"""
            ## 2. Load the IMDb Reviews Dataset

            The expected input file is `data/imdb_reviews.csv`. Common IMDb review files use columns such as `review` and `sentiment`, but the loader below also accepts close alternatives such as `text`, `label`, or `class`.
            """,
        ),
        notebook_cell(
            "code",
            r"""
            DATA_PATH = Path("data/imdb_reviews.csv")

            if not DATA_PATH.exists():
                raise FileNotFoundError(
                    "Place the IMDb dataset at data/imdb_reviews.csv. "
                    "Expected columns are review/text and sentiment/label."
                )

            raw_df = (
                spark.read
                .option("header", True)
                .option("inferSchema", True)
                .option("multiLine", True)
                .option("escape", '"')
                .csv(str(DATA_PATH))
            )

            print("Rows:", raw_df.count())
            raw_df.printSchema()
            raw_df.show(5, truncate=80)
            """,
        ),
        notebook_cell(
            "markdown",
            r"""
            ## 3. Standardize Columns and Inspect Class Balance

            This step renames the text and target columns to `review` and `label`. Positive sentiment is encoded as `1.0`; negative sentiment is encoded as `0.0`.
            """,
        ),
        notebook_cell(
            "code",
            r"""
            columns_lower = {c.lower(): c for c in raw_df.columns}

            text_candidates = ["review", "text", "content", "comment"]
            label_candidates = ["sentiment", "label", "class", "target"]

            text_col = next((columns_lower[c] for c in text_candidates if c in columns_lower), None)
            label_col = next((columns_lower[c] for c in label_candidates if c in columns_lower), None)

            if text_col is None or label_col is None:
                raise ValueError(
                    f"Could not identify text and label columns. Available columns: {raw_df.columns}"
                )

            df = raw_df.select(
                F.col(text_col).cast("string").alias("review"),
                F.col(label_col).alias("raw_label")
            ).dropna()

            df = df.withColumn(
                "label",
                F.when(F.lower(F.col("raw_label").cast("string")).isin("positive", "pos", "1", "true"), F.lit(1.0))
                 .when(F.lower(F.col("raw_label").cast("string")).isin("negative", "neg", "0", "false"), F.lit(0.0))
                 .otherwise(F.col("raw_label").cast(DoubleType()))
            ).dropna(subset=["label"])

            df = df.select("review", "label")
            df.cache()

            print("Clean rows:", df.count())
            df.groupBy("label").count().orderBy("label").show()
            """,
        ),
        notebook_cell(
            "code",
            r"""
            balance_pdf = df.groupBy("label").count().orderBy("label").toPandas()
            balance_pdf["sentiment"] = balance_pdf["label"].map({0.0: "Negative", 1.0: "Positive"})

            plt.figure(figsize=(6, 4))
            plt.bar(balance_pdf["sentiment"], balance_pdf["count"], color=["#D1495B", "#2A9D8F"])
            plt.title("IMDb Review Class Distribution")
            plt.xlabel("Sentiment")
            plt.ylabel("Number of reviews")
            plt.tight_layout()
            plt.show()
            """,
        ),
        notebook_cell(
            "markdown",
            r"""
            ## 4. Text Preprocessing

            Reviews often contain HTML tags, punctuation, numbers, mixed case, and extra spaces. The preprocessing below lowercases reviews, removes HTML tags, removes non-letter characters, and collapses whitespace. Tokenization and stopword removal are performed inside Spark ML pipeline stages so the same transformations are applied consistently during training and testing.
            """,
        ),
        notebook_cell(
            "code",
            r"""
            clean_df = (
                df.withColumn("clean_review", F.lower(F.col("review")))
                  .withColumn("clean_review", F.regexp_replace("clean_review", r"<[^>]+>", " "))
                  .withColumn("clean_review", F.regexp_replace("clean_review", r"[^a-zA-Z\s]", " "))
                  .withColumn("clean_review", F.regexp_replace("clean_review", r"\s+", " "))
                  .withColumn("clean_review", F.trim(F.col("clean_review")))
                  .filter(F.length("clean_review") > 0)
            )

            clean_df.select("review", "clean_review", "label").show(3, truncate=100)
            """,
        ),
        notebook_cell(
            "code",
            r"""
            length_df = clean_df.withColumn("word_count", F.size(F.split(F.col("clean_review"), " ")))
            length_pdf = length_df.select("word_count").sample(False, 0.2, seed=42).toPandas()

            plt.figure(figsize=(8, 4))
            plt.hist(length_pdf["word_count"], bins=40, color="#4E79A7", edgecolor="white")
            plt.title("Distribution of Review Lengths")
            plt.xlabel("Words per review")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()
            """,
        ),
        notebook_cell(
            "markdown",
            r"""
            ## 5. TF-IDF Feature Extraction

            The model pipeline uses:

            1. `RegexTokenizer` to split cleaned reviews into words.
            2. `StopWordsRemover` to remove common English stopwords.
            3. `CountVectorizer` to create term-frequency vectors.
            4. `IDF` to down-weight very common terms and produce TF-IDF features.
            """,
        ),
        notebook_cell(
            "code",
            r"""
            train_df, test_df = clean_df.randomSplit([0.8, 0.2], seed=42)
            train_df.cache()
            test_df.cache()

            print("Training rows:", train_df.count())
            print("Testing rows:", test_df.count())

            tokenizer = RegexTokenizer(
                inputCol="clean_review",
                outputCol="tokens",
                pattern=r"\W+",
                minTokenLength=2
            )

            stopwords = StopWordsRemover(
                inputCol="tokens",
                outputCol="filtered_tokens"
            )

            count_vectorizer = CountVectorizer(
                inputCol="filtered_tokens",
                outputCol="tf_features",
                vocabSize=50000,
                minDF=2.0
            )

            idf = IDF(
                inputCol="tf_features",
                outputCol="features"
            )
            """,
        ),
        notebook_cell(
            "markdown",
            r"""
            ## 6. Train PySpark ML Classification Models

            Three supervised classifiers are trained for comparison:

            - Logistic Regression
            - Naive Bayes
            - Decision Tree
            """,
        ),
        notebook_cell(
            "code",
            r"""
            models = {
                "Logistic Regression": LogisticRegression(
                    featuresCol="features",
                    labelCol="label",
                    maxIter=50,
                    regParam=0.01,
                    elasticNetParam=0.0
                ),
                "Naive Bayes": NaiveBayes(
                    featuresCol="features",
                    labelCol="label",
                    modelType="multinomial",
                    smoothing=1.0
                ),
                "Decision Tree": DecisionTreeClassifier(
                    featuresCol="features",
                    labelCol="label",
                    maxDepth=10,
                    seed=42
                )
            }

            fitted_models = {}
            predictions = {}

            for name, classifier in models.items():
                print(f"Training {name}...")
                pipeline = Pipeline(stages=[tokenizer, stopwords, count_vectorizer, idf, classifier])
                fitted_models[name] = pipeline.fit(train_df)
                predictions[name] = fitted_models[name].transform(test_df).cache()

            print("Training complete.")
            """,
        ),
        notebook_cell(
            "markdown",
            r"""
            ## 7. Model Evaluation

            Accuracy, weighted precision, weighted recall, and F1-score are computed using PySpark's `MulticlassClassificationEvaluator`. Area under ROC is included as an additional diagnostic when supported.
            """,
        ),
        notebook_cell(
            "code",
            r"""
            evaluators = {
                "Accuracy": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy"),
                "Precision": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision"),
                "Recall": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall"),
                "F1-score": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1"),
            }

            binary_auc = BinaryClassificationEvaluator(
                labelCol="label",
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            )

            results = []
            for name, pred_df in predictions.items():
                row = {"Model": name}
                for metric, evaluator in evaluators.items():
                    row[metric] = evaluator.evaluate(pred_df)
                try:
                    row["AUC"] = binary_auc.evaluate(pred_df)
                except Exception:
                    row["AUC"] = None
                results.append(row)

            results_pdf = pd.DataFrame(results).sort_values("F1-score", ascending=False)
            results_pdf
            """,
        ),
        notebook_cell(
            "code",
            r"""
            ax = results_pdf.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
                kind="bar",
                figsize=(10, 5),
                color=["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]
            )
            ax.set_title("Model Performance Comparison")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            ax.legend(loc="lower right")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.show()
            """,
        ),
        notebook_cell(
            "code",
            r"""
            best_model_name = results_pdf.iloc[0]["Model"]
            best_pred = predictions[best_model_name]
            print("Best model by F1-score:", best_model_name)

            confusion_pdf = (
                best_pred.groupBy("label", "prediction")
                .count()
                .orderBy("label", "prediction")
                .toPandas()
            )

            confusion_matrix = confusion_pdf.pivot(index="label", columns="prediction", values="count").fillna(0)
            confusion_matrix
            """,
        ),
        notebook_cell(
            "code",
            r"""
            plt.figure(figsize=(5, 4))
            plt.imshow(confusion_matrix.values, cmap="Blues")
            plt.title(f"Confusion Matrix: {best_model_name}")
            plt.xlabel("Predicted label")
            plt.ylabel("Actual label")
            plt.xticks(range(len(confusion_matrix.columns)), confusion_matrix.columns)
            plt.yticks(range(len(confusion_matrix.index)), confusion_matrix.index)

            for i in range(confusion_matrix.shape[0]):
                for j in range(confusion_matrix.shape[1]):
                    plt.text(j, i, int(confusion_matrix.values[i, j]), ha="center", va="center")

            plt.colorbar()
            plt.tight_layout()
            plt.show()
            """,
        ),
        notebook_cell(
            "markdown",
            r"""
            ## 8. Example Predictions and Error Review

            Error review is important because it shows where the sentiment classifier struggles. Misclassified reviews may include sarcasm, mixed sentiment, long plot summaries, or words whose meaning depends on context.
            """,
        ),
        notebook_cell(
            "code",
            r"""
            best_pred.select("clean_review", "label", "prediction", "probability").show(10, truncate=100)

            print("Misclassified examples:")
            best_pred.filter(F.col("label") != F.col("prediction")) \
                .select("clean_review", "label", "prediction", "probability") \
                .show(10, truncate=120)
            """,
        ),
        notebook_cell(
            "markdown",
            r"""
            ## 9. Findings

            In typical IMDb review experiments, Logistic Regression and Naive Bayes perform strongly because TF-IDF features capture discriminative sentiment words effectively. Decision Trees are easier to interpret but usually perform worse on sparse high-dimensional text features. The final conclusion should be based on the executed results table above after running the notebook on the full dataset.

            ## 10. Academic Integrity Statement

            This project uses an openly available IMDb movie review dataset for educational analysis. All preprocessing, model training, evaluation, visualization, and interpretation steps are documented transparently. Any external dataset or library used should be cited in the written report.
            """,
        ),
        notebook_cell(
            "code",
            r"""
            spark.stop()
            """,
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10+",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


README = r"""
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
"""


DATA_README = r"""
# Dataset Instructions

Place the IMDb Movie Reviews dataset in this folder as:

```text
imdb_reviews.csv
```

Recommended columns:

```text
review,sentiment
```

Example row:

```text
"This movie was surprisingly emotional and beautifully acted.",positive
```

The notebook also supports text column names such as `text`, `content`, or `comment`, and label column names such as `label`, `class`, or `target`.
"""


REPORT = r"""
# Sentiment Analysis of IMDb Movie Reviews Using Apache Spark and PySpark Machine Learning

**Course:** DASC6021 Big Data Analytics & Data Visualization  
**Project Weight:** 20%  
**Group Members:** Replace with group member names and IDs  
**Submission Date:** Replace with submission date

## Abstract

This project investigates sentiment analysis of IMDb movie reviews using Apache Spark and PySpark Machine Learning. The objective is to classify movie reviews into positive and negative sentiment categories using scalable big data analytics methods. The analysis uses a CSV version of the IMDb Movie Reviews dataset, loads it into a Spark DataFrame, performs text preprocessing, tokenization, stopword removal, TF-IDF feature extraction, and trains three supervised classifiers: Logistic Regression, Naive Bayes, and Decision Tree. The models are evaluated using accuracy, precision, recall, and F1-score, and the results are visualized through class distribution charts, review length distributions, model comparison charts, and confusion matrices. The project demonstrates how Spark can support distributed text analytics workflows and shows that linear probabilistic models generally perform well on sparse TF-IDF text features. The final notebook provides a reproducible implementation suitable for running on a local Spark environment or larger distributed cluster.

## 1. Introduction

Online movie reviews contain valuable information about audience opinions, preferences, and emotional reactions. Platforms such as IMDb host large volumes of user-generated text that can help researchers, producers, distributors, and recommendation systems understand public perception of films. However, manual analysis of thousands or millions of reviews is time-consuming and inconsistent. Sentiment analysis addresses this problem by using natural language processing and machine learning to automatically identify whether a text expresses a positive or negative opinion.

The aim of this project is to design and implement a scalable sentiment analysis pipeline for IMDb movie reviews using Apache Spark and PySpark. Spark is well suited to this task because it can process large datasets in distributed memory and provides high-level APIs for DataFrame operations, machine learning pipelines, and model evaluation. PySpark enables the implementation to be written in Python while still using Spark's distributed processing engine.

This project focuses on binary sentiment classification. Each review is represented as text and each label indicates positive or negative sentiment. The machine learning workflow includes environment setup, dataset loading, preprocessing, tokenization, stopword removal, feature extraction using TF-IDF, model training, evaluation, visualization, and interpretation. Three classification algorithms are compared: Logistic Regression, Naive Bayes, and Decision Tree. The comparison helps identify which model is most appropriate for sparse text features in a big data environment.

## 2. Problem Statement and Objectives

The problem addressed in this project is the automatic classification of IMDb movie reviews into positive and negative sentiment categories. Because review datasets can be large and text processing can be computationally expensive, the project uses Apache Spark rather than a single-machine workflow.

The main objectives are:

- To configure a PySpark environment for large-scale text analytics.
- To load the IMDb Movie Reviews dataset from CSV into a Spark DataFrame.
- To clean and transform raw review text into a machine-learning-ready format.
- To tokenize reviews and remove common stopwords.
- To extract numerical features using TF-IDF.
- To train Logistic Regression, Naive Bayes, and Decision Tree classifiers using PySpark ML.
- To evaluate models using accuracy, precision, recall, and F1-score.
- To visualize the dataset and model results.
- To explain the findings and limitations in a clear academic format.

## 3. Dataset Description

The project uses the IMDb Movie Reviews dataset, assumed to be available as `imdb_reviews.csv`. A typical version of this dataset contains review text and a sentiment label. The review column stores user-written movie reviews, while the sentiment column stores either positive or negative labels. The dataset is appropriate for sentiment analysis because the labels provide supervised learning targets and the reviews contain natural language expressions of opinion.

The notebook is designed to handle common column names such as `review`, `text`, `content`, or `comment` for the review field, and `sentiment`, `label`, `class`, or `target` for the target field. Positive reviews are encoded as `1.0`, and negative reviews are encoded as `0.0`. This encoding allows the models to treat the task as binary classification.

Initial exploratory analysis includes counting the number of rows, inspecting the schema, previewing sample reviews, and visualizing the class distribution. Class distribution is important because a highly imbalanced dataset can make accuracy misleading. For example, if most reviews were positive, a model could obtain high accuracy by predicting positive for most inputs while performing poorly on negative examples. Therefore, precision, recall, and F1-score are also used in evaluation.

## 4. Tools and Environment

The required tool for the project is Apache Spark with PySpark in a Jupyter Notebook environment. The implementation uses the following main components:

- Apache Spark for distributed data processing.
- PySpark SQL for loading, cleaning, and transforming structured data.
- PySpark ML for machine learning pipelines, feature extraction, classifiers, and evaluators.
- Jupyter Notebook for executable documentation.
- Matplotlib and pandas for visualizing aggregated outputs.

The Spark session is created inside the notebook using `SparkSession.builder`. The configuration sets the application name and adjusts shuffle partitions for a manageable local execution environment. On a larger cluster, the same notebook can be adapted by changing Spark configuration settings or submitting the job through a cluster manager.

## 5. Methodology

The methodology follows a standard big data machine learning pipeline.

### 5.1 Data Loading

The CSV file is loaded using Spark's CSV reader with header inference, schema inference, multiline support, and quote escaping. Multiline and escape options are important for movie reviews because review text may contain commas, quotation marks, and line breaks. After loading, the notebook prints the row count, schema, and sample rows.

### 5.2 Data Cleaning

The raw dataset is standardized into two columns: `review` and `label`. Missing values are removed. Review text is converted to lowercase, HTML tags are removed, non-letter characters are replaced with spaces, repeated whitespace is collapsed, and empty reviews are filtered out. Lowercasing reduces duplicate vocabulary entries caused by capitalization differences. Removing HTML and punctuation reduces noise before tokenization.

### 5.3 Tokenization

Tokenization splits cleaned review strings into individual words. The notebook uses PySpark's `RegexTokenizer`, which is suitable for large datasets because it operates as part of a distributed Spark ML pipeline. A minimum token length is used to remove very short fragments that are unlikely to provide useful sentiment information.

### 5.4 Stopword Removal

Stopwords are common words such as "the", "is", "and", and "of". These words usually occur frequently but often carry limited sentiment meaning. PySpark's `StopWordsRemover` removes common English stopwords from the tokenized review text. This reduces dimensionality and improves the quality of downstream feature extraction.

### 5.5 Feature Extraction Using TF-IDF

Machine learning models require numerical input, so the cleaned tokens must be transformed into feature vectors. The project uses TF-IDF, which combines term frequency and inverse document frequency. Term frequency captures how often a word appears in a review. Inverse document frequency reduces the weight of words that appear in many reviews and increases the weight of more distinctive words. The notebook uses `CountVectorizer` to generate term-frequency vectors and `IDF` to create TF-IDF features.

TF-IDF is appropriate for this project because sentiment classification often depends on discriminative words such as "excellent", "boring", "beautiful", "terrible", "waste", or "masterpiece". Although TF-IDF does not fully capture word order or sarcasm, it provides a strong baseline for scalable text classification.

### 5.6 Model Training

The dataset is split into training and testing subsets using an 80/20 split. Three models are trained:

Logistic Regression is a linear classifier that works well with high-dimensional sparse text features. It estimates the probability that a review belongs to the positive class and is often a strong baseline for text classification.

Naive Bayes is a probabilistic classifier that assumes feature independence. Although this assumption is simplified, Naive Bayes is efficient and often performs well on word-frequency and TF-IDF features.

Decision Tree is a non-linear model that splits data based on feature thresholds. It is interpretable but can struggle with sparse, high-dimensional text data because text vectors contain many possible terms and most values are zero.

Each model is trained inside a Spark ML `Pipeline` that includes tokenization, stopword removal, vectorization, IDF transformation, and classification. Pipelines make the workflow reproducible because the same transformations are applied to training and test data.

## 6. Evaluation Metrics

The models are evaluated using accuracy, precision, recall, and F1-score.

Accuracy measures the proportion of correct predictions among all predictions. It is easy to understand but can be misleading if the dataset is imbalanced.

Precision measures how many reviews predicted as positive are actually positive. High precision means the model avoids false positive errors.

Recall measures how many actual positive reviews are correctly identified. High recall means the model avoids false negative errors.

F1-score is the harmonic mean of precision and recall. It is useful when a balanced measure of classification performance is needed.

The notebook uses PySpark's `MulticlassClassificationEvaluator` to compute these metrics. It also calculates a confusion matrix for the best model to show true positives, true negatives, false positives, and false negatives. The confusion matrix provides more detailed insight into model behavior than a single score.

## 7. Visualization

Visualization supports interpretation and communication of findings. The notebook includes four main visualizations.

The class distribution bar chart shows how many positive and negative reviews are present. This helps determine whether the dataset is balanced.

The review length histogram shows the distribution of words per review. IMDb reviews often vary greatly in length, and this visualization helps explain text complexity.

The model comparison chart displays accuracy, precision, recall, and F1-score for each model. This makes it easy to compare classifier performance.

The confusion matrix heatmap shows where the best model makes correct and incorrect predictions. This is useful for discussing model strengths and weaknesses.

Because Spark DataFrames can be large, the notebook aggregates results in Spark first and then converts only small summary outputs to pandas for plotting. This follows good big data practice by avoiding unnecessary collection of full datasets into local memory.

## 8. Expected Findings and Discussion

The final findings should be based on the actual executed results after running the notebook on the full dataset. In many IMDb sentiment classification experiments, Logistic Regression performs strongly because linear models are effective on sparse TF-IDF vectors. Naive Bayes is also expected to perform competitively because it is designed for word-frequency-like inputs and is computationally efficient. Decision Tree is likely to produce lower performance because sparse text vectors have many dimensions and tree-based splits may not generalize as well without extensive tuning.

If Logistic Regression achieves the highest F1-score, the result suggests that weighted sentiment terms are sufficient to separate many positive and negative reviews. Words such as "excellent", "great", "amazing", and "wonderful" may contribute to positive classification, while words such as "bad", "boring", "awful", and "waste" may contribute to negative classification. However, TF-IDF cannot understand deeper language context. A review that says "not good" may be harder to classify because the sentiment depends on the relationship between words. Sarcasm, irony, mixed opinions, and long plot summaries can also cause errors.

The use of Spark is valuable because the same workflow can scale from a local sample to a larger distributed dataset. Operations such as tokenization, vectorization, and model training are expressed as distributed transformations. This is important in big data analytics because text datasets can become too large for conventional single-machine processing.

## 9. Limitations

The project has several limitations. First, TF-IDF treats documents as bags of words and ignores word order. This means phrases such as "not bad" and "bad" may not be represented with enough contextual difference. Second, stopword removal may remove words that are important for negation if not handled carefully. Third, Decision Tree models may not be ideal for sparse text data without additional feature selection or tuning. Fourth, model performance depends on the quality and balance of the dataset. If the dataset contains noisy labels or duplicate reviews, results may be affected.

Another limitation is that the project focuses on binary sentiment classification. Real-world opinions can be neutral, mixed, or multi-dimensional. A review may praise acting but criticize plot, pacing, or visual effects. Future work could extend the task to multi-class sentiment, aspect-based sentiment analysis, or deep learning approaches using word embeddings and transformer models.

## 10. Conclusion

This project demonstrates a complete big data analytics workflow for sentiment analysis of IMDb movie reviews using Apache Spark and PySpark. The notebook loads a large text dataset into a Spark DataFrame, preprocesses raw review text, tokenizes reviews, removes stopwords, extracts TF-IDF features, trains three classification models, evaluates them using standard metrics, and presents visualizations for interpretation.

The project shows that PySpark ML pipelines are effective for scalable text classification. Logistic Regression and Naive Bayes are expected to be strong models for TF-IDF features, while Decision Tree provides a useful comparison but may be less suitable for high-dimensional sparse text. The final model choice should be based on the executed F1-score and confusion matrix. Overall, the project satisfies the assignment requirements by combining big data processing, machine learning, visualization, and academic reporting in a reproducible workflow.

## 11. Academic Integrity and Originality Statement

This submission is prepared as an original academic project for DASC6021 Big Data Analytics & Data Visualization. The IMDb review dataset is used for educational purposes and should be cited appropriately. The notebook, report, and slides document the complete analytical process so that results can be reproduced. Any external libraries, datasets, or references used in the final submission should be acknowledged by the group.

## References

Apache Software Foundation. (n.d.). *Apache Spark documentation*. https://spark.apache.org/docs/latest/

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies*, 142-150.

PySpark Documentation. (n.d.). *PySpark MLlib guide*. https://spark.apache.org/docs/latest/ml-guide.html
"""


SLIDES = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IMDb Sentiment Analysis with PySpark</title>
  <style>
    :root {
      --ink: #17202a;
      --muted: #506070;
      --accent: #2a9d8f;
      --warm: #e76f51;
      --gold: #f4a261;
      --paper: #f7f9fb;
      --line: #d8e1e8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--paper);
      overflow: hidden;
    }
    .deck {
      height: 100vh;
      width: 100vw;
      display: flex;
      transition: transform 250ms ease;
    }
    section {
      min-width: 100vw;
      height: 100vh;
      padding: 5vh 7vw;
      display: grid;
      align-content: center;
      gap: 24px;
      background: linear-gradient(180deg, #ffffff 0%, #eef5f7 100%);
    }
    h1 {
      font-size: 54px;
      line-height: 1.05;
      margin: 0;
      max-width: 980px;
      letter-spacing: 0;
    }
    h2 {
      font-size: 42px;
      margin: 0;
      letter-spacing: 0;
    }
    p, li {
      font-size: 24px;
      line-height: 1.4;
      color: var(--muted);
    }
    ul { margin: 0; padding-left: 30px; }
    .kicker {
      color: var(--accent);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1px;
      font-size: 16px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 18px;
    }
    .card {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 24px;
      min-height: 150px;
    }
    .card strong {
      font-size: 28px;
      display: block;
      margin-bottom: 10px;
    }
    .flow {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
      align-items: stretch;
    }
    .step {
      background: #fff;
      border-top: 8px solid var(--accent);
      border-radius: 8px;
      padding: 18px;
      font-size: 20px;
      color: var(--ink);
      min-height: 120px;
    }
    .bar {
      height: 34px;
      background: var(--accent);
      margin: 12px 0;
      color: #fff;
      padding: 6px 12px;
      font-weight: 700;
      border-radius: 4px;
    }
    .bar.orange { background: var(--warm); width: 88%; }
    .bar.gold { background: var(--gold); width: 78%; color: var(--ink); }
    .bar.green { width: 96%; }
    .footer {
      position: fixed;
      bottom: 18px;
      left: 7vw;
      right: 7vw;
      display: flex;
      justify-content: space-between;
      color: #687887;
      font-size: 14px;
    }
    @media (max-width: 800px) {
      h1 { font-size: 36px; }
      h2 { font-size: 30px; }
      p, li { font-size: 18px; }
      section { padding: 5vh 6vw; }
      .grid, .flow { grid-template-columns: 1fr; }
      .card { min-height: auto; }
    }
  </style>
</head>
<body>
  <main class="deck" id="deck">
    <section>
      <div class="kicker">DASC6021 Big Data Analytics & Data Visualization</div>
      <h1>Sentiment Analysis of IMDb Movie Reviews Using Apache Spark and PySpark Machine Learning</h1>
      <p>Group project presentation | Replace with group names</p>
    </section>
    <section>
      <h2>Project Goal</h2>
      <p>Classify IMDb movie reviews into positive and negative sentiment categories using a scalable PySpark machine learning workflow.</p>
      <div class="grid">
        <div class="card"><strong>Input</strong><p>Large CSV file of movie review text and sentiment labels.</p></div>
        <div class="card"><strong>Processing</strong><p>Spark DataFrames, cleaning, tokenization, stopword removal, TF-IDF.</p></div>
        <div class="card"><strong>Output</strong><p>Model comparison using accuracy, precision, recall, and F1-score.</p></div>
      </div>
    </section>
    <section>
      <h2>Why Spark?</h2>
      <ul>
        <li>Distributed processing for large text datasets.</li>
        <li>DataFrame API for scalable cleaning and aggregation.</li>
        <li>ML pipelines for reproducible feature extraction and classification.</li>
        <li>Same notebook can run locally or on a cluster.</li>
      </ul>
    </section>
    <section>
      <h2>Data Pipeline</h2>
      <div class="flow">
        <div class="step">Load CSV into Spark DataFrame</div>
        <div class="step">Clean review text and encode labels</div>
        <div class="step">Tokenize and remove stopwords</div>
        <div class="step">Extract TF-IDF features</div>
        <div class="step">Train and evaluate models</div>
      </div>
    </section>
    <section>
      <h2>Models Compared</h2>
      <div class="grid">
        <div class="card"><strong>Logistic Regression</strong><p>Strong baseline for sparse TF-IDF features.</p></div>
        <div class="card"><strong>Naive Bayes</strong><p>Fast probabilistic model for text classification.</p></div>
        <div class="card"><strong>Decision Tree</strong><p>Interpretable model used as a non-linear comparison.</p></div>
      </div>
    </section>
    <section>
      <h2>Evaluation Metrics</h2>
      <ul>
        <li><strong>Accuracy:</strong> overall percentage of correct predictions.</li>
        <li><strong>Precision:</strong> reliability of positive predictions.</li>
        <li><strong>Recall:</strong> ability to find actual positive reviews.</li>
        <li><strong>F1-score:</strong> balanced measure of precision and recall.</li>
      </ul>
    </section>
    <section>
      <h2>Expected Result Pattern</h2>
      <div class="bar green">Logistic Regression: strongest expected TF-IDF performance</div>
      <div class="bar orange">Naive Bayes: competitive and efficient baseline</div>
      <div class="bar gold">Decision Tree: interpretable but weaker on sparse text</div>
      <p>Replace this slide with actual metric values after running the notebook.</p>
    </section>
    <section>
      <h2>Key Insights</h2>
      <ul>
        <li>TF-IDF highlights words that distinguish positive and negative reviews.</li>
        <li>Linear models are usually well suited to high-dimensional text vectors.</li>
        <li>Errors may come from sarcasm, negation, mixed opinions, and plot-heavy reviews.</li>
      </ul>
    </section>
    <section>
      <h2>Limitations and Future Work</h2>
      <ul>
        <li>Bag-of-words features do not fully capture word order or context.</li>
        <li>Future work could test n-grams, feature selection, hyperparameter tuning, or transformer embeddings.</li>
        <li>The workflow can scale to larger datasets using a Spark cluster.</li>
      </ul>
    </section>
    <section>
      <h2>Conclusion</h2>
      <p>The project delivers a complete PySpark sentiment analysis pipeline for IMDb movie reviews, including scalable preprocessing, TF-IDF features, three machine learning classifiers, evaluation metrics, and visual results.</p>
    </section>
  </main>
  <div class="footer">
    <span>Use left/right arrows to navigate</span>
    <span id="counter">1 / 10</span>
  </div>
  <script>
    const deck = document.getElementById('deck');
    const slides = Array.from(deck.children);
    const counter = document.getElementById('counter');
    let index = 0;
    function showSlide(next) {
      index = Math.max(0, Math.min(slides.length - 1, next));
      deck.style.transform = `translateX(${-index * 100}vw)`;
      counter.textContent = `${index + 1} / ${slides.length}`;
    }
    document.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowRight' || event.key === ' ') showSlide(index + 1);
      if (event.key === 'ArrowLeft') showSlide(index - 1);
    });
    showSlide(0);
  </script>
</body>
</html>
"""


REQUIREMENTS = r"""
pyspark>=3.5.0
jupyter>=1.0.0
pandas>=2.0.0
matplotlib>=3.7.0
"""


SAMPLE_SCRIPT = r"""
from pathlib import Path
import csv


rows = [
    ("A moving and beautifully acted film with excellent pacing.", "positive"),
    ("The story was boring and the characters felt completely flat.", "negative"),
    ("Wonderful direction, strong performances, and a memorable ending.", "positive"),
    ("I wanted to like it, but the plot was messy and dull.", "negative"),
    ("A charming movie that balances humor and emotion very well.", "positive"),
    ("The film wasted a good idea with weak writing and poor editing.", "negative"),
]

path = Path("data/imdb_reviews.csv")
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["review", "sentiment"])
    writer.writerows(rows)

print(f"Wrote sample dataset to {path}. Replace it with the full IMDb dataset for final results.")
"""


def main():
    write(PROJECT / "README.md", README)
    write(PROJECT / "data" / "README.md", DATA_README)
    write(PROJECT / "reports" / "Report_IMDb_Sentiment_PySpark.md", REPORT)
    write(PROJECT / "slides" / "index.html", SLIDES)
    write(PROJECT / "requirements.txt", REQUIREMENTS)
    write(PROJECT / "scripts" / "create_sample_dataset.py", SAMPLE_SCRIPT)

    notebook = build_notebook()
    nb_path = PROJECT / "notebooks" / "Sentiment_Analysis_IMDb_PySpark.ipynb"
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nb_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

    print(f"Generated project at {PROJECT}")


if __name__ == "__main__":
    main()
