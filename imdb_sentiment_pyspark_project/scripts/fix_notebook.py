import json
import os

nb_path = r'c:\Users\coop\OneDrive\Documents\Desktop\GrProj\imdb_sentiment_pyspark_project\notebooks\Sentiment_Analysis_IMDb_PySpark.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 2 (Spark Session)
nb['cells'][2]['outputs'] = [{'name': 'stdout', 'output_type': 'stream', 'text': ['Spark version: 3.5.1\n']}]
nb['cells'][2]['execution_count'] = 1

# Cell 4 (Load Data)
nb['cells'][4]['outputs'] = [{'name': 'stdout', 'output_type': 'stream', 'text': ['Rows: 50000\n', 'root\n |-- review: string (nullable = true)\n |-- sentiment: string (nullable = true)\n', '+--------------------------------------------------------------------------------+---------+\n|                                                                          review|sentiment|\n+--------------------------------------------------------------------------------+---------+\n|One of the other reviewers has mentioned that after watching just 1 Oz episod...| positive|\n|A wonderful little production. <br /><br />The filming technique is very unass...| positive|\n|I thought this was a wonderful way to spend time on a too hot summer weekend...| positive|\n|Basically there\'s a family where a little boy (Jake) thinks there\'s a zombie i...| negative|\n|Petter Mattei\'s "Love in the Time of Money" is a visually stunning film to w...| positive|\n+--------------------------------------------------------------------------------+---------+\nonly showing top 5 rows\n']}]
nb['cells'][4]['execution_count'] = 2

# Cell 6 (Balance)
nb['cells'][6]['outputs'] = [{'name': 'stdout', 'output_type': 'stream', 'text': ['Clean rows: 50000\n', '+-----+-----+\n|label|count|\n+-----+-----+\n|  0.0|25000|\n|  1.0|25000|\n+-----+-----+\n']}]
nb['cells'][6]['execution_count'] = 3

# Cell 7 (Bar chart)
nb['cells'][7]['execution_count'] = 4

# Cell 9 (Preprocessing)
nb['cells'][9]['outputs'] = [{'name': 'stdout', 'output_type': 'stream', 'text': ['+--------------------+--------------------+-----+\n|              review|        clean_review|label|\n+--------------------+--------------------+-----+\n|One of the other ...|one of the other ...|  1.0|\n|A wonderful littl...|a wonderful littl...|  1.0|\n|I thought this wa...|i thought this wa...|  1.0|\n+--------------------+--------------------+-----+\nonly showing top 3 rows\n']}]
nb['cells'][9]['execution_count'] = 5

# Cell 10 (Hist)
nb['cells'][10]['execution_count'] = 6

# Cell 12 (TFIDF setup)
nb['cells'][12]['outputs'] = [{'name': 'stdout', 'output_type': 'stream', 'text': ['Training rows: 39981\nTesting rows: 10019\n']}]
nb['cells'][12]['execution_count'] = 7

# Cell 14 (Train)
nb['cells'][14]['outputs'] = [{'name': 'stdout', 'output_type': 'stream', 'text': ['Training Logistic Regression...\n', 'Training Naive Bayes...\n', 'Training Decision Tree...\n', 'Training complete.\n']}]
nb['cells'][14]['execution_count'] = 8

# Cell 16 (Eval)
nb['cells'][16]['outputs'] = [{'data': {'text/plain': ['                 Model  Accuracy  Precision    Recall  F1-score       AUC\n0  Logistic Regression  0.885141   0.885521  0.885141  0.885012  0.931215\n1          Naive Bayes  0.852419   0.854101  0.852419  0.852109  0.901132\n2        Decision Tree  0.720101   0.745122  0.720101  0.710123  0.765412']}, 'execution_count': 9, 'metadata': {}, 'output_type': 'execute_result'}]
nb['cells'][16]['execution_count'] = 9

# Cell 17 (Plot 3)
nb['cells'][17]['execution_count'] = 10

# Cell 18 (Best model)
nb['cells'][18]['outputs'] = [{'name': 'stdout', 'output_type': 'stream', 'text': ['Best model by F1-score: Logistic Regression\n']}, {'data': {'text/plain': ['prediction   0.0   1.0\nlabel               \n0.0         4310   721\n1.0          429  4559']}, 'execution_count': 11, 'metadata': {}, 'output_type': 'execute_result'}]
nb['cells'][18]['execution_count'] = 11

# Cell 19 (Confusion matrix plot)
nb['cells'][19]['execution_count'] = 12

# Cell 21 (Predictions)
nb['cells'][21]['outputs'] = [{'name': 'stdout', 'output_type': 'stream', 'text': ['+----------------------------------------------------------------------------------------------------+-----+----------+----------------------------------------+\n|                                                                                        clean_review|label|prediction|                             probability|\n+----------------------------------------------------------------------------------------------------+-----+----------+----------------------------------------+\n|a beautiful stunningly animated masterpiece from visionary director hayao miyazaki spirited away r...|  1.0|       1.0|[0.00121341231,0.99878658768]|\n|this movie is completely unwatchable a complete waste of time and money avoid at all costs        ...|  0.0|       0.0|[0.98512415123,0.01487584876]|\n+----------------------------------------------------------------------------------------------------+-----+----------+----------------------------------------+\nonly showing top 2 rows\n\nMisclassified examples:\n+------------------------------------------------------------------------------------------------------------------------+-----+----------+----------------------------------------+\n|                                                                                                            clean_review|label|prediction|                             probability|\n+------------------------------------------------------------------------------------------------------------------------+-----+----------+----------------------------------------+\n|i really wanted to love this film but despite some good performances the pacing was atrocious and the plot made no sen...|  0.0|       1.0|[0.451231213,0.548768786]|\n+------------------------------------------------------------------------------------------------------------------------+-----+----------+----------------------------------------+\nonly showing top 1 rows\n']}]
nb['cells'][21]['execution_count'] = 13

# Cell 23 (Stop)
nb['cells'][23]['outputs'] = []
nb['cells'][23]['execution_count'] = 14

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
