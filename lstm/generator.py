import random as rd
import pandas as pd
import numpy as np
amount = []
for i in range(325):
   amount.append(rd.randint(3, 10) + 11)

df = pd.read_csv("export_dataframe_nodeA.csv")
x = df.iloc[ :, 0:1].values
date = np.reshape(x, len(x))

df = pd.DataFrame(list(zip(date, amount)), columns = ['Date', 'amount'])
print(df)
df.to_csv(r'/home/akrom/fabric-samples/myBlockchain/application/thesis.csv', index=False)