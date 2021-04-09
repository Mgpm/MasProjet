import pyspark.ml.evaluation as ev
class selectFeature:
    def __init__(self,df):
        self.df = df


    def selectModel(self,m1,m2):
         r=[],
         for model in [m1,m2]:
            df_train,df_test = self.df.randomSplit([0.7,0.3],seed=666)
            result = model.fit(df_train).transform(df_test).select('labels','prediction')
            eval = ev.MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='labels')
            eval.evaluate(result,{eval.metricName:'accuracy'})
            r.append([model,eval.evaluate(result,{eval.metricName:'accuracy'})])
         if r[0][1] > r[1][1]:
             self.m= m1
         else:
             self.m=m2

    def featureSelect(self,pd):
        dt = self.m.fit(self.df)
        self.featureImp = ["var"+str(i) for i in dt.featureImportances.indices]
        self.featureImpAndLabel = ["var"+str(i) for i in dt.featureImportances.indices]+["labels","label"]
        self.featureImportances = pd.Series(index=self.featuresImp,data=dt.featureImportances.values)

    def numberFeatureImportances(self):
        return self.featureImportances.shape

    def headFeatureImportances(self):
        self.dfImp = self.df.select(self.featureImpAndLabel)
        return self.dfImp.toPandas().head()

    def plotFeatureImportances(self,plt):
        self.featureImportances.sort_values().plot(kind='barh',figsize=(5,50),title='Feature Imporatance')
        plt.show()







