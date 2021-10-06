import os
from pyspark.sql import SparkSession

os.environ['SPARK_HOME'] = 'C:\Spark\spark-3.0.3-bin-hadoop2.7'


def init_spark():
    spark = SparkSession.builder.appName("BigData").getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def main():
    spark, sc = init_spark()
    nums = sc.parallelize(range(1, 1000))
    print(nums.filter(lambda x: x % 2 == 1).map(lambda x: x * x).collect())


if __name__ == '__main__':
    main()
