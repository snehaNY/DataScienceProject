import pandas as pd
import numpy as np
import os

def read_data():
    raw_data_path = os.path.join(os.path.pardir,'data','raw');
    train_data_path = os.path.join(raw_data_path,'train.csv')
    test_data_path = os.path.join(raw_data_path,'test.csv')
    #Read data with all default parameters
    train_df = pd.read_csv(train_data_path, index_col='PassengerId')
    test_df = pd.read_csv(test_data_path, index_col='PassengerId')
    test_df['Survived'] = -888
    df = pd.concat((train_df,test_df),axis=0)
    return df

def process_data(df):
    #using the method chaining concept
    return(df
           #Create the title attribute
           .assign(Title = lambda x : x.Name.map(get_title))
           #working with missing values
           .pipe(fill_missing_values)
           #create fare bin feature
           .assign(FareBin = lambda x : pd.qcut(x.Fare,4,labels=['very_low','low','high','very_high']))
           #create age state
           .assign(AgeState = lambda x: np.where(x.Age >= 18,'Adult','Child'))
           .assign(FamilySize = lambda x: x.Parch + x.SibSp +1)
           .assign(IsMother = lambda x: np.where
                   (((x.Sex =='female') & (x.Parch > 0) & (x.Age > 18) & (x.Title != 'Miss')),1,0))
           #create deck feature
           .assign(Cabin = lambda x : np.where(x.Cabin == 'T',np.nan,x.Cabin))
           .assign(Deck = lambda x: x.Cabin.map(get_deck))
           #feature encoding
           .assign(IsMale = lambda x: np.where(x.Sex== 'male',1,0))
           .pipe(pd.get_dummies,columns=['Deck','Pclass','Title','FareBin','Embarked','AgeState'])
           #drop unnecessary columns
           .drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'],axis=1)
           .pipe(reorder_columns)
           
    )
def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0],'Z')
    
def reorder_columns(df):
    columns = [column for column in df.columns if column!='Survived']
    columns = ['Survived']+columns
    df = df[columns]
    return df
    
def fill_missing_values(df):
    #embarked
    df.Embarked.fillna('C',inplace=True)
    #fare
    median_fare = df[(df.Pclass ==3) & (df.Embarked =='S')]['Fare'].median()
    df.Fare.fillna(median_fare,inplace=True)
    #age
    title_median_age = df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_median_age,inplace=True)
    return df

def write_data(df):
    processed_data_path = os.path.join(os.path.pardir,'data','processed')
    write_train_path = os.path.join(processed_data_path,'train.csv')
    write_test_path = os.path.join(processed_data_path,'test.csv')
    df[df.Survived != -888].to_csv(write_train_path)
    #test data
    columns = [column for column in df.columns if column!='Survived']
    df[df.Survived == -888][columns].to_csv(write_test_path)
    
def get_title(name):
    dt_title = {
        'mr':'Mr',
        'miss':'Miss',
        'mrs':'Mrs',
        'master':'Master',
        'don':'Sir',
        'rev':'Sir',
        'dr':'Mr',
        'mme':'Mrs',
        'ms':'Mrs',
        'major':'Mr',
        'lady':'Miss',
        'sir':'Sir',
        'mlle':'Miss',
        'col':'Mr',
        'capt':'Mr',
        'the countess':'Mrs',
        'jonkheer':'Sir',
        'dona':'Miss'
    }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return dt_title[title]

if __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    write_data(df)
