from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler as v,PCA,StandardScaler
from pyspark.ml.clustering import KMeans as km
from pyspark.sql import functions as func, types as t
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from loadDataVector import loadDataVector
from selectFeature import  selectFeature
from clustering import clusters
from dataAnalysis import dataAnalysis
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").appName("projetRCP216").getOrCreate()




