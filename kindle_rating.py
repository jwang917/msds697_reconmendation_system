from pyspark import SparkContext, SparkConf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import asc, desc
from pyspark.ml.feature import StringIndexer

conf = SparkConf().setAppName("project")\
                  .set('spark.executor.memory', '12G')\
                  .set('spark.driver.memory', '10G')\
                  .set('spark.executor.core', 6)\
                  .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
                  .set("spark.default.parallelism", 6)
sc = SparkContext(conf=conf)
sc.setLogLevel("OFF")
ss = SparkSession.builder.getOrCreate()

kindle_raw = sc.textFile("./Data/ratings_Kindle_Store.csv")
kindle_ratings = kindle_raw.map(lambda x: x.split(','))\
                           .map(lambda x: Row(userId=str(x[0]), itemId=str(x[1]),
                                          rating=float(x[2])))

ratings_df = ss.createDataFrame(kindle_ratings).persist()

def indexStringColumns(df, cols):
    newdf = df
    
    for c in cols:
        si = StringIndexer(inputCol=c, outputCol=c+"-num")
        sm = si.fit(newdf) 
        newdf = sm.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+"-num", c)
    return newdf

ratings_df = indexStringColumns(ratings_df, ["userId", "itemId"]).persist()
train, valid, test = ratings_df.randomSplit([0.7, 0.1, 0.2], 10)

from pyspark.ml.evaluation import RegressionEvaluator

sc.setCheckpointDir('checkpoint/')
evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating',
                                predictionCol='prediction')

def compute_rmse(model, data):
    predictions = model.transform(data)
    rmse = evaluator.evaluate(predictions)
    print("Root mean square = ", str(rmse))
    return rmse

als = ALS(maxIter=3, regParam=0.05, numUserBlocks=10, numItemBlocks=10, 
          implicitPrefs=False, alpha=1.0, userCol='userId', itemCol='itemId',
          seed=1, ratingCol='rating', nonnegative=True, coldStartStrategy="drop",
          checkpointInterval=10, intermediateStorageLevel="MEMORY_AND_DISK",
          finalStorageLevel="MEMORY_AND_DISK")

model = als.fit(train)
print(compute_rmse(model, valid))
