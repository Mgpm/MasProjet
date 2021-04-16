
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

    def clusterViz(self,n,df): # affiche le graphique des clusters
        kmean = self.km(k=n,featuresCol="features",predictionCol="prediction",initMode="random")
        cluster_df = kmean.fit(df).transform(df)
        self.cluster_df = cluster_df.toPandas()
        self.cluster_df.head()

    def selectOneVarByClass(self,k,pd):# Selection une seul valeur de chaque class
        clusters = pd.DataFrame(index=range(self.cluster_df.shape[0]))
        for i in range(k):
            clusters['class' + str(i)] = pd.Series(self.cluster_df[self.cluster_df['prediction'] == i]['index'].values)
        clusterVar = clusters.dropna()
        clusterVar = clusterVar.reset_index()
        self.clusterVar = clusterVar.drop('index',axis=1)







