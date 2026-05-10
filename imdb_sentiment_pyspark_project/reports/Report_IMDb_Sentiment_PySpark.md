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

## 10. Reproducibility and Implementation Notes

Reproducibility is an important requirement in academic data analytics projects. The notebook is designed so that the same sequence of operations can be executed from top to bottom after the dataset is placed in the correct folder. The input path is fixed as `data/imdb_reviews.csv`, and the random train-test split uses a fixed seed. This means the split should remain consistent across repeated runs, provided the dataset remains the same.

The implementation uses Spark ML pipelines to reduce accidental differences between training and testing transformations. Without a pipeline, a common error is to fit text preprocessing on the full dataset or apply inconsistent transformations to the test data. In this project, the tokenizer, stopword remover, count vectorizer, IDF transformer, and classifier are combined in one pipeline. The pipeline is fitted only on training data, and the fitted transformations are then applied to test data. This approach supports a more realistic evaluation because the model does not learn vocabulary statistics from the test set before prediction.

The notebook also caches selected DataFrames after cleaning and splitting. Caching is useful in Spark because model training and evaluation may reuse the same data multiple times. By caching training and test sets, Spark can avoid recomputing earlier transformations repeatedly. On a small local machine, this improves execution time. On a cluster, caching can also reduce repeated disk reads and shuffle overhead, although memory availability should always be considered.

The visualization strategy follows a scalable pattern. Large Spark DataFrames are not converted directly into pandas DataFrames. Instead, the notebook first aggregates data in Spark, such as class counts or confusion matrix counts, then converts only the small aggregated result to pandas for plotting. This is important because collecting a full large dataset to the driver can cause memory errors and defeats the purpose of using Spark.

## 11. Ethical Considerations

Sentiment analysis systems can be useful, but they should be interpreted carefully. A model trained on IMDb reviews learns patterns from a specific platform, time period, and user population. It may not generalize perfectly to reviews from other websites, languages, cultures, or domains. For example, words and expressions used in film criticism may differ from those used in product reviews or social media posts. Therefore, the results should be presented as classification outcomes for the dataset used, not as a universal measure of human opinion.

There is also a risk of over-trusting automated sentiment labels. A review may contain complex feelings, irony, cultural references, or mixed evaluation. A binary positive or negative label can simplify that complexity. In real applications, automated predictions should support human judgment rather than replace it completely, especially when decisions affect creators, workers, or communities.

The project also follows academic integrity principles by documenting each analytical step and identifying the dataset source in the references. The notebook is transparent enough for another student or instructor to inspect the code, rerun the analysis, and verify the results. Any changes made by group members should be recorded honestly, and the final submitted notebook should contain executed outputs from the actual dataset.

## 12. Group Work Plan

The project can be divided into clear responsibilities for group submission. One member can handle environment setup, Spark session configuration, and dataset loading. A second member can focus on preprocessing, tokenization, stopword removal, and TF-IDF feature extraction. A third member can train and tune the machine learning models. A fourth member can prepare evaluation charts, interpret results, and refine the report and presentation.

During final preparation, all group members should review the notebook and confirm that it runs successfully from a clean start. The group should also replace placeholder names in the report and slides, insert the actual results table after executing the notebook, and export the report to the instructor's required format. This shared review process helps avoid inconsistent results between the notebook, report, and presentation.

## 13. Recommendations for Improvement

Several improvements could be explored if more time or computing resources were available. First, the group could test n-gram features so the model can capture short phrases such as "not good", "very bad", or "highly recommend". Second, hyperparameter tuning could be added using Spark's `CrossValidator` or `TrainValidationSplit`. Parameters such as vocabulary size, minimum document frequency, regularization strength, and tree depth could be optimized systematically.

Third, feature selection could be used to reduce dimensionality and improve efficiency. High-dimensional TF-IDF vectors may contain rare terms that add noise. Filtering features or adjusting `minDF` can reduce model size and training time. Fourth, the group could compare TF-IDF with modern embedding-based features. Word embeddings and transformer models can capture semantic context more effectively than bag-of-words features, although they require more computing resources and may be less straightforward to run in a basic Spark notebook.

Finally, the group could perform a deeper error analysis by sampling false positives and false negatives. This would reveal whether errors are mostly caused by sarcasm, negation, review length, ambiguous wording, or label noise. Error analysis is useful because it turns evaluation metrics into practical understanding of model behavior.

## 14. Conclusion

This project demonstrates a complete big data analytics workflow for sentiment analysis of IMDb movie reviews using Apache Spark and PySpark. The notebook loads a large text dataset into a Spark DataFrame, preprocesses raw review text, tokenizes reviews, removes stopwords, extracts TF-IDF features, trains three classification models, evaluates them using standard metrics, and presents visualizations for interpretation.

The project shows that PySpark ML pipelines are effective for scalable text classification. Logistic Regression and Naive Bayes are expected to be strong models for TF-IDF features, while Decision Tree provides a useful comparison but may be less suitable for high-dimensional sparse text. The final model choice should be based on the executed F1-score and confusion matrix. Overall, the project satisfies the assignment requirements by combining big data processing, machine learning, visualization, and academic reporting in a reproducible workflow.

## 15. Academic Integrity and Originality Statement

This submission is prepared as an original academic project for DASC6021 Big Data Analytics & Data Visualization. The IMDb review dataset is used for educational purposes and should be cited appropriately. The notebook, report, and slides document the complete analytical process so that results can be reproduced. Any external libraries, datasets, or references used in the final submission should be acknowledged by the group.

## References

Apache Software Foundation. (n.d.). *Apache Spark documentation*. https://spark.apache.org/docs/latest/

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies*, 142-150.

PySpark Documentation. (n.d.). *PySpark MLlib guide*. https://spark.apache.org/docs/latest/ml-guide.html
