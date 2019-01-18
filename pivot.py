from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *


sc = SparkContext.getOrCreate()
ss = SparkSession.builder.config('spark.sql.pivotMaxValues', '9999').getOrCreate()

def convert_rating(row):
    row[2] = float(row[2])
    return row

path = '/Users/shijialiang/data/amazon/ratings_Video_Games.csv'
games_data = sc.textFile(path).map(lambda x: x.split(',')).map(lambda x: convert_rating(x))

schema = StructType([StructField("user", StringType(), False),
                      StructField("item", StringType(), False),
                      StructField("rating", FloatType(), False),
                      StructField("time", StringType(), True)
                    ])

games_df = ss.createDataFrame(games_data, schema).cache()
games_df.write.saveAsTable('games')

ss.sql('select * from games\
        where user in (select distinct user from games group by user having count(item) > 10)\
          and item in (select distinct item from games group by item having count(user) > 10)')\
        .groupBy('item').pivot('user').avg('rating')\
        .write.saveAsTable('filtered_games')

