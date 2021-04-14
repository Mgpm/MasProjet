
class dataAnalysis:
    def __init__(self,class_df):
        self.class_df = class_df

    def selectData(self,df_init):
        self.df = df_init.select(self.class_df.values[0,:])



