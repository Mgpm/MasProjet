
class dataAnalysis:
    def __init__(self,class_df):
        self.class_df = class_df

    def selectData(self,df_init,pd):
        colsSelect = [i for i in self.class_df.values[0,:]]+["label"]
        self.df = df_init.select(colsSelect)
        self.dfExp = self.df.toPandas()

    def head(self):
        self.dfExp.head()

    def tail(self):
        self.dfExp.tail()

    def columns(self):
        print(self.dfExp.columns)

    def shape(self):
        print(self.dfExp.shape)


    def info(self):
        self.dfExp.info()

    def describe(self):
        self.dfExp[self.dfExp.columns[:-1]].describe()

    def HistoVarNum(self,plt):
        self.dfExp(column=self.dfExp[:-1])
        plt.show()

    def BarVarCat(self,plt):
        self.dfExp['label'].value_counts().plot(kind='bar',title='Target Variable')
        plt.show()

    def Correlation(self,sns,plt):
        cor_mat = self.dfExp.corr()
        sns.heatmap(cor_mat,annot=True)
        plt.show()











