Sparkling Titanic
=================

### Introduction

`titanic_logReg.py` trains a logistic regression and makes prediction for [Titanic dataset](http://kaggle.com/c/titanic/data) as part of Kaggle competition using Apache-Spark [spark-1.3.1-bin-hadoop2.4](http://spark.apache.org/downloads.html) and its Python API on a local machine. I used `pyspark_csv.py` to load data as Spark DataFrame, for more instructions see [this](http://github.com/seahboonsiew/pyspark-csv). 

The following will be added later

*   Imputing NAs in train and test sets
*   Cross-validation
*   Using more features and feature engineering
*   RandomForest classifier, SVM, etc.

### Loading PySpark Script

Run `$SPARK_HOME/bin/spark-submit scriptDirectoryPath/titanic_logReg.py`. For multithreading, you can add option `--master local[N]` where N is the number of threads.




