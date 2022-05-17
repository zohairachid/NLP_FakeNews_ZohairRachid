# Basic libraries
import pandas as pd
import joblib

def clean_data(dataframe):
    
    # Drop duplicate rows
    dataframe.drop_duplicates(subset='title', inplace=True)
    
    # Remove punctation
    dataframe['title'] = dataframe['title'].str.replace('[^\w\s]',' ')

    # Remove numbers 
    dataframe['title'] = dataframe['title'].str.replace('[^A-Za-z]',' ')

    # Make sure any double-spaces are single 
    dataframe['title'] = dataframe['title'].str.replace('  ',' ')
    dataframe['title'] = dataframe['title'].str.replace('  ',' ')

    # Transform all text to lowercase
    dataframe['title'] = dataframe['title'].str.lower()
    
    print("New shape:", dataframe.shape)
    return dataframe.head()


def preprocess (query):
    a={'title': [query]}
    df_a1=pd.DataFrame(data=a)
    clean_data(df_a1)
    df_a1 = df_a1.reset_index(drop=True)
    X_df_a1 = df_a1['title']    
    cvec = joblib.load(open('vectorizer.pkl', 'rb'))
    Xcvec_df_a1 = cvec.transform(X_df_a1)
    return Xcvec_df_a1