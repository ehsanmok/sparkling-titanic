#    Copyright (C) 2015 Ehsan Mohyedin Kermani
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#    Contact: ehsanmo1367@gmail.com, ehsanmok@cs.ubc.ca

from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *
from pyspark.sql.functions import UserDefinedFunction
from pyspark.mllib.linalg import Vectors
import os
import pyspark_csv as pycsv

os.environ["SPARK_LOCAL_IP"] = "127.0.1.1" # set local IP

DATADIR = "./Python/PySpark/Titanic/data" # data directory

def mySparkContext():
    """
    Sets the Spark Context
    """
    conf = (SparkConf()
            .setMaster("local")
            .setAppName("Titanic Logistic Regression")
            .set("spark.executor.memory", "1g"))
    sc = SparkContext(conf = conf)
    return sc

sc = mySparkContext()
sqlCtx = SQLContext(sc)
sc.addPyFile("Spark/spark-1.3.1-bin-hadoop2.4/pyspark_csv.py")

def loadDF(filename):
    """
    Load and parse filename as pyspark.sql.DataFrame
    using pyspark_csv.py
    """
    path = os.path.join(DATADIR, filename)
    plain = sc.textFile(path)
    df = pycsv.csvToDataFrame(sqlCtx, plain, sep=',')
    return df

# DataFrames are immutable. Must transform RDD to another RDD for binary sex
udf = UserDefinedFunction(lambda x: 1 if x == "male" else 0, IntegerType())

def sex_to_bin(df):
    """
    Maps male to 1 and female to 0
    """
    df = df.select(*[udf(column).alias('Sex') \
    if column == 'Sex' else column for column in df.columns])
    return df

if __name__ == "__main__":
    
    train = loadDF("train.csv")
    test = loadDF("test.csv")

    testPassengerId = test.select('PassengerId').map(lambda x: x.PassengerId)

    train = train.select('Survived', 'Pclass', 'Sex', 'SibSp', 'Parch')
    test = test.select('Pclass', 'Sex', 'SibSp', 'Parch')

    train = sex_to_bin(train)
    test = sex_to_bin(test)

    print "number of men in train and test resp. : %d, %d" \
        %(train.select('Sex').map(lambda x: x.Sex).sum() \
        ,test.select('Sex').map(lambda x: x.Sex).sum())

    # format train for Logistic Regression as (label, features)
    ntrain = train.map(lambda x: Row(label = float(x[0]) \
         ,features = Vectors.dense(x[1:]))).toDF().cache() # Logistic Regression is iterative, need caching
    ntest = test.map(lambda x: Row(features = Vectors.dense(x[0:]))).toDF()
    
    lr = LogisticRegression(maxIter = 100, regParam = 0.1)
    model = lr.fit(ntrain)
    pred = model.transform(ntest).select('prediction').map(lambda x: x.prediction)
    
    # configure the submission format as follows
    submit = sqlCtx.createDataFrame(testPassengerId.zip(pred), ["PassengerId", "Survived"])
    ## NOTE: rdd1.zip(rdd2) works provided that both RDDs have the same partitioner and the same number 
    # of elements per partition, otherwise should either repartition or can do:
    # submit = sqlCtx.createDataFrame(pred.zipWithIndex().map(lambda x: (x[1]+892L, x[0])), ["PassengerId", "Survived"])
    # Side: 891L is the number of training samples
    os.chdir(DATADIR)
    # file is small so can save pandas.DataFrame as csv
    submit.toPandas().to_csv("prediction.csv", index = False)
    # if not, should saveAsTextFile:
    # submit.rdd.saveAsTextFile("/home/ehsan/Python/PySpark/Titanic/data/prediction")
    sc.stop()

 
 
 



