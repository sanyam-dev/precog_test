import pandas as pd

class Utils:
    @staticmethod
    def load_data():
        DF = dict()
        index = 0
        for i in range(1, 101):
            index = '00' + str(i)
            index = index[-3:]
            file_path = "../data/anonymized_data/Asset_" + index+'.csv'
            df = pd.DataFrame(pd.read_csv(file_path))
            df.set_index('Date', inplace=True)
            DF[i] = df
        return pd.concat(DF.values(), axis=1, keys=['Asset_' + str(i) for i in DF.keys()])
    
    @staticmethod
    def split_train_test(DF, test_size=0.2):
        trdf_ ={}
        tedf_ ={}
        for i, df in DF.items():
            n_test = int(len(df) * test_size)
            trdf_[i] = df.iloc[:-n_test]
            tedf_[i] = df.iloc[-n_test:]
        return trdf_, tedf_
