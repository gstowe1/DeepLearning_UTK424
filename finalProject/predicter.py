import numpy as np

class PREDICTER:
    def __init__(self,df,model,generator):
        self.df = df
        self.model = model
        self.gen = generator


    def norm_series(self, series):
        norm_min = self.df[self.gen].min()
        norm_max = self.df[self.gen].max()

        series = (series-norm_min) / (norm_max - norm_min)

        return series


    def unNorm_series(self,series):
        norm_min = self.df[self.gen].min()
        norm_max = self.df[self.gen].max()

        series = series * (norm_max - norm_min) + norm_min

        return series


    def predict(self,series):

        series = np.reshape(self.norm_series(series),(-1,12,1))

        pred = self.model.predict(series, verbose = 0)[0]

        return self.unNorm_series(pred)
        