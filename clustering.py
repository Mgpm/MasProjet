
class clusters:

    def __init__(self,k,plt):
        self.km=k
        self.plt = plt
    def clusterElbow(self,df,i,n):
        errors=[]
        for k in range(i,n):
            kmean = self.km(k=k,featuresCol="features")
            model = kmean.fit(df)
            intra_distance = model.computeCost(df)
            errors.append(intra_distance)
        cluster_num = range(i,n)
        self.plt.figure(figsize=(15,5))
        self.plt.xlabel('Numbres de clusters')
        self.plt.ylabel('SSE')
        self.plt.plot(cluster_num,errors)

    def clusterViz(self,n,df):
        kmean = self.km(k=n,featuresCol="features",predictionCol="prediction",initMode="random")
        model = self.km.fit(df)
        predictions = model.transform(df)
        cluster_df = predictions.toPandas().head(2)


