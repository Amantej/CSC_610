import pandas as pd
import matplotlib.pyplot as plt
for i in range(0,10):
    df = pd.read_csv('../data/csvfile/assign3/subdataset'+str(i)+'.csv',header = None)[1:]
    print(df.describe())
    #plt.hist(df)