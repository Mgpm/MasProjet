from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler as v,PCA, StandardScaler
from pyspark.sql import functions as func, types as t
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from loadDataVector import loadDataVector
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").appName("projetRCP216").getOrCreate()
lo = loadDataVector(spark,t)
lo.schema(0,10000)
lo.load("data/data.arff")



