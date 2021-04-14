import pyspark.ml.evaluation as ev

class selectFeature: #selectionne des variables suivants leur importances
    def __init__(self,df,model):
        self.df = df
        self.model = model

    def featureSelect(self,pd): # determine les variables importantes par le model choisi
        dt = self.model.fit(self.df)
        self.featureImp = ["var"+str(i) for i in dt.featureImportances.indices]
        self.featureImpAndLabel = ["var"+str(i) for i in dt.featureImportances.indices]+["labels","label"]
        self.featureImportances = pd.Series(index=self.featureImp,data=dt.featureImportances.values)


    def dataFeatureImportances(self,df): #crée une nouvelle dataFrame avec des variables importantes selectionnées a partir du df initial
        self.dfImp = df.select(self.featureImp)


    def plotFeatureImportances(self,plt): # affichage par graphe
        self.featureImportances.sort_values().plot(kind='barh',figsize=(5,50),title='Feature Importance')
        plt.show()







