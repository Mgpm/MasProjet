
class loadDataVector: # Cette classe s'occupe de charges les données et les transforment en vecteur
    def __init__(self,sp,t,func):
        self.spark = sp
        self.t = t
        self.func = func

    def schema(self,i,n): # methode de generation de schéma
        self.sch = self.t.StructType([self.t.StructField('var'+str(i),self.t.DoubleType(),True) for i in range(n)]).add('label',self.t.StringType(),True)

    def load(self,f): # transforme les données en dataframe spark
        self.df = self.spark.read.format('csv').schema(self.sch).option('sep',',').load(f)

    def vectAssembler(self,v): # vectorise le dataframe obtenu avec codage de la variable label
        vect = v(inputCols=self.df.columns[:-1],outputCol='features').transform(self.df).select('features','label')
        pdList = list(vect.select('label').distinct().toPandas().values)
        transfo = self.func.udf(lambda x:pdList.index(x))
        self.vectdf = vect.select('features','label').withColumn('labels',transfo('label').cast(self.t.DoubleType()))


    def vectAssemblerFeatureImp(self,vect,v): # obtention du dataframe vectorisé pour la reduction par ordre de complexité
        dataTranspose = vect.toPandas()
        dataTranspose = dataTranspose.transpose()
        dataTranspose = dataTranspose.reset_index()
        sch1 = self.t.StructType([self.t.StructField('index',self.t.StringType(),True)]+[self.t.StructField(str(i),self.t.DoubleType(),True) for i in range(1500)])
        dataTransp = self.spark.createDataFrame(dataTranspose,schema=sch1)
        self.vectTransposed = v(inputCols=dataTransp.columns[1:],outputCol="features").transform(dataTransp)
