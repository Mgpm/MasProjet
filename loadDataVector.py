
class loadDataVector:
    def __init__(self,sp,t,func):
        self.spark = sp
        self.t = t
        self.func = func

    def schema(self,i,n):
        self.sch = self.t.StructType([self.t.StructField('var'+str(i),self.t.DoubleType(),True) for i in range(n)]).add('label',self.t.StringType(),True)

    def load(self,f):
        self.df = self.spark.read.format('csv').schema(self.sch).option('sep',',').load(f)

    def vectAssembler(self,v):
        vect = v().setInputCols([self.df.columns[:-1]]).setOutputCol('features').transform(self.df).select('features','label')
        pdList = list(vect.select('label').distinct().toPandas().value)
        transfo = self.func.udf(lambda x:pdList.index(x))
        vectdf = vect.select('features','label').withColumn('labels',transfo('label').cast(self.t.DoubleType()))
        return vectdf

