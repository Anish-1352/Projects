import pandas as pd
import numpy as np
def load_data():
    factor_df = pd.read_csv("C:/Users/anish/OneDrive/Desktop/Columbia MAFN/Intro to Math Finance/Quantitative Deep Learning Model (Fama-French 3-Factor)/3 Fama Factors.csv", skiprows=0)

    factor_df.rename(columns={'Date': 'Date', 'Mkt-RF':'Mkt-RF','SMB':'SMB','HML':'HML','RF':'RF'}, inplace=True)

    factor_df['Date'] = pd.to_datetime(factor_df['Date'], format='%Y%m%d')


    msft_df = pd.read_csv("C:/Users/anish/OneDrive/Desktop/Columbia MAFN/Intro to Math Finance/Quantitative Deep Learning Model (Fama-French 3-Factor)/MSFT data for FF.csv", skiprows=0)

    msft_df.rename(columns={'Date':'Date', 'Adj Close':'Adj CLose', 'daily returns':'daily returns'}, inplace=True)
    msft_df['Date']= pd.to_datetime(msft_df['Date'], format='%d-%m-%Y')
    msft_df = msft_df.loc[:, ~msft_df.columns.str.contains('^Unnamed')]


    model_data = pd.merge(factor_df, msft_df, on="Date", how="inner")

    model_data.dropna(inplace=True)

    model_data.set_index('Date', inplace=True)
    
    return model_data

    
if __name__ =="__main__":
    model_data = load_data()
    print(model_data.head())
    
    




