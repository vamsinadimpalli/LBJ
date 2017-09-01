# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 08:59:11 2017

@author:  -- Vamsi Nadimpalli
"""

import pandas as pd 
import numpy as np   
import pytz  
import glob
import gc

from datetime import datetime, timedelta
from scipy.spatial import distance
from scipy.stats import linregress
from itertools import cycle  
from pytz import timezone 
from sklearn.model_selection import KFold, cross_val_score, train_test_split 



# Load Monthly Customer Data
print('Loading Monthly Customer Data...')
Time1 = datetime.now()  # time point 1 
path =r'C:/Users/c584g/Documents/Projects/LBJ&NTE/LBJ/Monthly Customer Data' # use your path
allFiles = glob.glob(path + "/*.csv") 
Cust_Month_Summ = pd.DataFrame()
list_ = []
# create a dataframe by looping through all the files in this folder
for file in allFiles:
    Year        = int(file.replace(r'.csv', '').split('_')[3].split('-')[0])  
    Month       = int(file.replace(r'.csv', '').split('_')[3].split('-')[1])
    Year_Month  = str(int(file.replace(r'.csv', '').split('_')[3].split('-')[0])*100+int(file.replace(r'.csv', '').split('_')[3].split('-')[1]))
    Month_Idx   = allFiles.index(file)
    #extracting month from the file name to add it as a column in the merged frame
    df          = pd.read_csv(file,  usecols=["Tag_ID", "First_Seen", "Last_Seen", "Txn_Count",
                                  "Txn_Cost","Workday_Txn_Count","Workday_Txn_Cost",
                                  "Peak_Txn_Count","Peak_Txn_Cost" , "Trip_Count",
                                  "Trip_Cost","Workday_Trip_Count","Workday_Trip_Cost",
                                  "Peak_Trip_Count","Peak_Trip_Cost","Vehicle_Type"] ,
                            header=0) 
    df['Month']     = Year_Month
    df['Month_Idx'] = Month_Idx
    df['First_Seen']     = pd.to_datetime(df['First_Seen']) 
    df['First_Seen_Month'] = df['First_Seen'].dt.year*100+df['First_Seen'].dt.month
    df['First_Seen_Month'] = df['First_Seen_Month'].astype(str)
    df['New_Cust_Flag'] = np.where(df['Month'] == df['First_Seen_Month'], 'Brand New', 'Existing')
    list_.append(df)
Cust_Month_Summ = pd.concat(list_) 
print('Time taken Loading = ' + str( (datetime.now() - Time1).total_seconds()/60)+' mins ')
# Cust_Month_Summ.head(3) 


del list_  # cleaning up intermediate objects
del df 
gc.collect()



#Create a aggregated customer level summary excluding the latest month data ( 7 months effectively)
Time2                   = datetime.now() 
Cust_Grouped_Hist       = Cust_Month_Summ[Cust_Month_Summ.Month_Idx < (len(allFiles) - 1)].groupby(['Tag_ID','Vehicle_Type'], as_index= False) 
print(Cust_Grouped_Hist.head(3))
Cust_Summ               = Cust_Grouped_Hist.agg({'Txn_Count' : np.sum, 
                                       'Month': 'count',
                                       'Txn_Cost' : np.sum, 
                                       'Peak_Txn_Count' : np.sum ,  
                                       'Workday_Txn_Count' : np.sum ,   
                                       'Trip_Count' : np.sum,  
                                       'Trip_Cost' : np.sum,
                                       'Peak_Trip_Count' : np.sum ,  
                                       'Workday_Trip_Count' : np.sum  }, sort= False) 
Cust_Summ.rename(columns={'Txn_Count': 'Total_Txn', 
                           'Month'   : 'Months_Showed_Up',
                           'Txn_Cost': 'Total_Cost',
                           'Peak_Txn_Count': 'Total_Peak_Txns',
                           'Workday_Txn_Count': 'Total_Workday_Txns', 
                           'Trip_Count': 'Total_Trip_Count',
                           'Trip_Cost': 'Total_Trip_Cost',
                           'Peak_Trip_Count': 'Total_Peak_Trips',
                           'Workday_Trip_Count': 'Total_Workday_Trips' }, inplace=True)

Cust_Summ['Avg_Txns']       = Cust_Summ.Total_Txn/Cust_Summ.Months_Showed_Up
Cust_Summ['Perc_Peak']      = Cust_Summ.Total_Peak_Txns/Cust_Summ.Total_Txn
# Binning Avg Monthly Txn Values
Monthly_Txn_Bin_Values      = [0, 5, 10, 15, 25, 50, 10000]
Monthly_Txn_Bin_Names       = ['1: Very Low', '2: Low', '3: Medium' , '4: High', '5: Very High', '6: Ext High'] 
Cust_Summ['Avg_Monthly_Txn_Bin']  = pd.cut(Cust_Summ['Avg_Txns'], Monthly_Txn_Bin_Values, labels=Monthly_Txn_Bin_Names)
# Binning % Peak Values
Perc_Peak_Bin_Values        = [-1, 0, 0.25, 0.5, 0.75, 0.999999, 1.0]
Perc_Peak_Bin_Names         = ['1: Peak = 0%', '2: Peak <= 25%', '3: Peak <= 50%','4: Peak <= 75%','5: Peak < 99.99%' , '6: Peak = 100%'] 
Cust_Summ['Perc_Peak_Bin']  = pd.cut(Cust_Summ['Perc_Peak'], Perc_Peak_Bin_Values, labels=Perc_Peak_Bin_Names)
#print(Cust_Summ.head(3))

#Create a customer summary for the last month which indicates the future revenue 
Cust_Grouped_Future    = Cust_Month_Summ[Cust_Month_Summ.Month_Idx == (len(allFiles) - 1)][['Tag_ID', 'Month_Idx','Txn_Cost', 'Txn_Count']]
#print(Cust_Grouped_Future.head(3))
Cust_Grouped_Future.rename(columns={'Txn_Cost': 'Txn_Cost_Future',  
                           'Txn_Count': 'Txn_Count_Future' }, inplace=True) 


Cust_Grouped_Recent    = Cust_Month_Summ[Cust_Month_Summ.Month_Idx == (len(allFiles) - 2)][['Tag_ID', 'Month_Idx','Txn_Cost', 'Txn_Count']]
Cust_Grouped_Recent.rename(columns={'Txn_Cost': 'Txn_Cost_Recent',  
                           'Txn_Count': 'Txn_Count_Recent' }, inplace=True) 
    
print('Time taken for Summarizing = ' + str( (datetime.now() - Time2).total_seconds()/60)+' mins ')

print('Creating a csv with Customer Summary for 7th Month')
Cust_Grouped_Future.to_csv("C:/Users/c584g/Documents/Projects/LBJ&NTE/LBJ/LBJ Loyalty Attributes/Cust_Final_Month_Summ.csv", sep=',',index= False)
print('Time taken for csv writing = ' + str( (datetime.now() - Time2).total_seconds()/60)+' mins ')
print('Done.')

print('Creating a csv with Customer Summary for latest month (6th month)')
Cust_Grouped_Recent.to_csv("C:/Users/c584g/Documents/Projects/LBJ&NTE/LBJ/LBJ Loyalty Attributes/Cust_Recent_Month_Summ.csv", sep=',',index= False)
print('Time taken for csv writing = ' + str( (datetime.now() - Time2).total_seconds()/60)+' mins ')
print('Done.')

#Merging customer level summary to customer month level summary
Cust_Month_Summ_merged              = pd.merge(Cust_Month_Summ[Cust_Month_Summ.Month_Idx < (len(allFiles) - 1)], Cust_Summ, on = 'Tag_ID') 
Cust_Month_Summ_merged              = Cust_Month_Summ_merged.sort_values(by=['Tag_ID', 'Month'])
Cust_Month_Summ_merged['Rank']      = Cust_Month_Summ_merged.groupby(['Tag_ID'])['Month'].rank(ascending=True) 


#Controlling for Seasonality in trips/txns
Month_Summ                = Cust_Month_Summ[Cust_Month_Summ.Month_Idx < (len(allFiles) - 1)].groupby(['Month'], as_index= False).agg({'Txn_Count' : np.sum,}, sort= False).sort_values(by=['Month'])
Month_Summ.rename(columns={'Txn_Count': 'Total_Month_Txn',  }, inplace=True) 
Month_Summ['Seasonality']           = Month_Summ.Total_Month_Txn/Month_Summ.Total_Month_Txn.sum()  
Cust_Month_Summ_merged              = pd.merge(Cust_Month_Summ_merged, Month_Summ, on = 'Month', how = 'left')  
Cust_Month_Summ_merged['Txn_Count_Adj'] = Cust_Month_Summ_merged['Txn_Count']/Cust_Month_Summ_merged['Seasonality']
temp = Cust_Month_Summ_merged.groupby(['Tag_ID'], as_index=False).agg({'Txn_Count_Adj' : np.sum,}, sort= False)
temp.rename(columns={'Txn_Count_Adj': 'Total_Txn_Count_Adj',  }, inplace=True) 
Cust_Month_Summ_merged              = pd.merge(Cust_Month_Summ_merged, temp, on = 'Tag_ID') 
Cust_Month_Summ_merged['Perc_Txns'] = Cust_Month_Summ_merged['Txn_Count_Adj']/Cust_Month_Summ_merged['Total_Txn_Count_Adj']
Cust_Month_Summ_merged['Perc_Rev']  = Cust_Month_Summ_merged['Txn_Cost']/Cust_Month_Summ_merged['Total_Cost'] 
Cust_Month_Summ_merged.Perc_Rev     = Cust_Month_Summ_merged.Perc_Rev.fillna(0)


#Estimating txn growth and revenue growth trend slopes per customer
Time3            = datetime.now()
Txn_growth_Slope = Cust_Month_Summ_merged[Cust_Month_Summ_merged.Months_Showed_Up>=3].groupby('Tag_ID').apply(lambda x: linregress(x.Rank,x.Perc_Txns)[0])
Txn_growth_Slope = Txn_growth_Slope.to_frame().reset_index().rename(columns= {0: 'Txn_Growth_Slope'})
Rev_Growth_Slope = Cust_Month_Summ_merged[Cust_Month_Summ_merged.Months_Showed_Up>=3].groupby('Tag_ID').apply(lambda x: linregress(x.Rank,x.Perc_Rev)[0])
Rev_Growth_Slope = Rev_Growth_Slope.to_frame().reset_index().rename(columns= {0: 'Rev_Growth_Slope'})
print('Time taken for slope estimation = ' + str( (datetime.now() - Time3).total_seconds()/60)+' mins ')


Cust_Month_Summ_merged_2                  = pd.merge(Cust_Month_Summ_merged, Rev_Growth_Slope, on = 'Tag_ID',how='left')
Cust_Month_Summ_merged_2                  = pd.merge(Cust_Month_Summ_merged_2, Txn_growth_Slope, on = 'Tag_ID',how='left')
Cust_Month_Summ_merged_2.Txn_Growth_Slope = Cust_Month_Summ_merged_2.Txn_Growth_Slope.fillna(-999)
Cust_Month_Summ_merged_2.Rev_Growth_Slope = Cust_Month_Summ_merged_2.Rev_Growth_Slope.fillna(-999)

Cust_Summ_Trimmed =  pd.merge(Cust_Summ[['Tag_ID','Avg_Monthly_Txn_Bin','Perc_Peak_Bin','Months_Showed_Up', 'Total_Txn', 'Total_Cost','Vehicle_Type']], Txn_growth_Slope, on = 'Tag_ID',how='left')
Cust_Summ_Trimmed.Txn_Growth_Slope = Cust_Summ_Trimmed.Txn_Growth_Slope.fillna(-999)

# Binning Avg Monthly Txn Values
Txn_Growth_Bin_Values = [-99999, -0.04, -0.01, 0.01, 0.04, 999999]
Txn_Growth_Bin_Names  = ['1:Fast Decline', '2:Slow Decline', '3: Stable' , '4: Slow Growth', '5: Fast Growth'] 
Cust_Summ_Trimmed['Txn_Growth_Bin']   = pd.cut(Cust_Summ_Trimmed['Txn_Growth_Slope'], Txn_Growth_Bin_Values, labels=Txn_Growth_Bin_Names)
Cust_Summ_Trimmed['Months_Showed_Up'] = Cust_Summ_Trimmed['Months_Showed_Up'].astype('category')

Cust_Summ_Trimmed  = pd.merge(Cust_Summ_Trimmed,Cust_Grouped_Future[['Tag_ID','Txn_Cost_Future','Txn_Count_Future']],on = 'Tag_ID',how='left' )
Cust_Summ_Trimmed.Txn_Cost = Cust_Summ_Trimmed.Txn_Cost_Future.fillna(0.0, inplace = True)  #filling NaN with zeros
Cust_Summ_Trimmed.Txn_Cost = Cust_Summ_Trimmed.Txn_Count_Future.fillna(0.0, inplace = True)  #filling NaN with zeros

                                                                      
def func(x):
    if x <=0.0:
        return 'NO'
    else: return 'YES'  
Cust_Summ_Trimmed['Retained'] = Cust_Summ_Trimmed['Txn_Count_Future'].apply(func)

train, test = train_test_split(Cust_Summ_Trimmed, test_size = 0.2) 
train['Train_Test_Flag'] = 'TRAIN'  
test['Train_Test_Flag']  = 'TEST'  
Cust_Summ_Trimmed = pd.concat([train,test]) 
#Cust_Month_Summ_merged_2.head(3)
Time4           = datetime.now()
print('Creating a csv with merged cust summary data')
Cust_Summ_Trimmed.to_csv("C:/Users/c584g/Documents/Projects/LBJ&NTE/LBJ/LBJ Loyalty Attributes/Cust_Summ.csv", sep=',',index= False)
print('Time taken for csv writing = ' + str( (datetime.now() - Time4).total_seconds()/60)+' mins ')
print('Done.')


file = "C:/Users/c584g/Documents/Projects/LBJ&NTE/LBJ/LBJ Loyalty Attributes/Cust_Summ.csv"
df          = pd.read_csv(file,  header=0) 

Cust_Summ_Trimmed  = pd.merge(df,Cust_Grouped_Recent[['Tag_ID','Txn_Cost_Recent','Txn_Count_Recent']],on = 'Tag_ID',how='left' )
Cust_Summ_Trimmed.Txn_Cost = Cust_Summ_Trimmed.Txn_Cost_Recent.fillna(0.0, inplace = True)  #filling NaN with zeros
Cust_Summ_Trimmed.Txn_Cost = Cust_Summ_Trimmed.Txn_Count_Recent.fillna(0.0, inplace = True)  #filling NaN with zeros

Time5           = datetime.now()
print('Creating a csv with merged cust summary data')
Cust_Summ_Trimmed.to_csv("C:/Users/c584g/Documents/Projects/LBJ&NTE/LBJ/LBJ Loyalty Attributes/Cust_Summ.csv", sep=',',index= False)
print('Time taken for csv writing = ' + str( (datetime.now() - Time5).total_seconds()/60)+' mins ')
print('Done.')

                                                                      
file = "C:/Users/c584g/Documents/Projects/LBJ&NTE/LBJ/LBJ Loyalty Attributes/Cust_Summ.csv"
df          = pd.read_csv(file, usecols=["Avg_Monthly_Txn_Bin", "Perc_Peak_Bin", "Months_Showed_Up", "Txn_Growth_Bin",
                                  "Vehicle_Type" ] , header=0)                                                                       
                                                                      
Pred_Df = df.drop_duplicates()
Pred_Df['Seg_ID'] = Pred_Df.index
Time5           = datetime.now()
print('Creating a csv with Segment combinations to be predicted')
Pred_Df.to_csv("C:/Users/c584g/Documents/Projects/LBJ&NTE/LBJ/LBJ Loyalty Attributes/Segs_Pred.csv", sep=',',index= False)
print('Time taken for csv writing = ' + str( (datetime.now() - Time5).total_seconds()/60)+' mins ')
print('Done.')


print('Creating a csv with merged cust summary data')
Cust_Month_Summ_merged_2.to_csv("C:/Users/c584g/Documents/Projects/LBJ&NTE/LBJ/LBJ Loyalty Attributes/Cust_Month_Summ_Merged.csv", sep=',',index= False)
print('Time taken for csv writing = ' + str( (datetime.now() - Time4).total_seconds()/60)+' mins ')
print('Done.')

del Cust_Month_Summ_merged, Cust_Month_Summ_merged_2, Cust_Grouped_Future, Cust_Grouped_Hist, 
del Cust_Summ, Cust_Summ_Trimmed
del Cust_Month_Summ
gc.collect()






