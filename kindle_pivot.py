from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate()
ss = SparkSession.builder.getOrCreate()

def convert_rating(row):
    row[2] = float(row[2])
    return row

path = '/Users/shijialiang/data/amazon/ratings_Kindle_Store.csv'
books_data = sc.textFile(path).map(lambda x: x.split(',')).map(lambda x: convert_rating(x)).cache()

from pyspark.sql.types import *
schema = StructType([StructField("user", StringType(), False),
                      StructField("item", StringType(), False),
                      StructField("rating", FloatType(), False),
                      StructField("time", StringType(), True)
                    ])

books_df = ss.createDataFrame(books_data, schema).cache()
books_df.write.saveAsTable('kindle')

ss.sql('select * from kindle\
        where user in (select distinct user from kindle group by user having count(item) > 100)\
          and item in (select distinct item from kindle group by item having count(user) > 60)')\
    .groupBy('item').pivot('user').avg('rating')\
    .write.saveAsTable('pivot_kindle')