import pandas as pd 
def main(nameNode):
    data = pd.read_csv("test-data.csv")
    #print(data['traderId'])
    data1 = data.loc[data['traderId'] == nameNode]
    data2 = data1.sort_values(by=['date'])
    dates = data2.iloc[:,2]
    #dates = dates.str.split("-", n = 2, expand = True)[2].astype(int)
    amount = data2.iloc[:,1]
    df = pd.DataFrame(list(zip(dates, amount)), columns=['dates', 'amount'])
    #get_data('test-data.csv')
    #dates.sort()
    df.to_csv (r'/home/akrom/fabric-samples/myBlockchain/application/export_dataframe_'+nameNode+'.csv', index = None, header=True)
main('nodeA')