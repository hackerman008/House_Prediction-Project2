import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics


#training using linear regression
def train1(df):

    x = df.ix[:,0:-1].values
    y = df.ix[:,'SalePrice'].values
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)

    lr_model = LinearRegression()
    lr_model.fit(x_train,y_train)
    
    prediction = lr_model.predict(x_test)
    print('Linear Regression MSE=',metrics.mean_squared_error(y_test,prediction))

#training using decision tree regressor
def train2(df):
    x = df.ix[:,0:-1].values
    y = df.ix[:,'SalePrice'].values
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)

    dtr_model = DecisionTreeRegressor(max_depth = 8,min_samples_split = 30)
    dtr_model.fit(x_train,y_train)
    
    prediction = dtr_model.predict(x_test)
    print('Decision tree regressor MSE=',metrics.mean_squared_error(y_test,prediction))
    
    return

def train3(df):
    x = df.ix[:,0:-1].values
    y = df.ix[:,'SalePrice'].values
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)
    
    n_features = 46
    rfr_model = RandomForestRegressor(n_estimators = 100,max_features = n_features,max_depth = None,min_samples_split = 2,random_state=1)
    rfr_model.fit(x_train,y_train)
    
    prediction = rfr_model.predict(x_test)
    print('Decision tree regressor MSE=',metrics.mean_squared_error(y_test,prediction))
    return
    
    
def train4(df,df2):
    x = df.ix[:,0:-1].values
    y = df.ix[:,'SalePrice'].values
    
    id1 = df2.ix[:,'Id']
    x_test = df2.ix[:,0:]
    
    n_features = 46
    rfr_model = RandomForestRegressor(n_estimators = 200,max_features = n_features,max_depth = None,min_samples_split = 2,n_jobs = -1,random_state =1)
    rfr_model.fit(x,y)
    
    prediction = rfr_model.predict(x_test)
    dataframe = pd.DataFrame(prediction,id1,columns=['SalePrice'])
    dataframe.to_csv('solution1.csv',index_label = 'Id')    
    return rfr_model,prediction 


def load_data(): 
    os.chdir(r'C:\programming\kaggle\house prediction dataset')
    df = pd.read_csv('train.csv')
    df2 = pd.read_csv('test.csv')
    return df,df2

def finding_null_columns(df,df2):
    l = []
    l2 = []
    a = df.isnull().any()
    b = df2.isnull().any()

    for i in range(0,81):
        if a[i] == True:
            l.append(a.index[i])
    
    for i in range(0,80):
        if b[i] == True:
            l2.append(a.index[i])
        
    return l,l2 
    
def removing_columns(l,l2,df,df2):
    labels_to_remove = list(set(l+l2))
    df = df.drop(axis = 1,labels = labels_to_remove)
    df2 = df2.drop(axis = 1,labels = labels_to_remove)
    return df,df2    
    

def converting_to_indicator_variables(df,df2):
    for i in df2.columns:
        if df2[i].dtype not in [np.int64,np.float64]:
            l_encoder = preprocessing.LabelEncoder()
            df[i] = l_encoder.fit_transform(df[i])
            
            l2_encoder = preprocessing.LabelEncoder()
            df2[i] = l2_encoder.fit_transform(df2[i])
    
    return df,df2

 
if __name__ == '__main__':
    #loading data    
    df,df2 = load_data()
    #finding columns with any nan values
    l,l2 = finding_null_columns(df,df2)
    #removing columns with nan values
    
    df,df2 = removing_columns(l,l2,df,df2)
    df,df2 = converting_to_indicator_variables(df,df2)
    
    train1(df)
    train2(df)
    train3(df)
    rfr_model,prediction = train4(df,df2)
    print('feature_importance = \n',rfr_model.feature_importances_)
    
    









