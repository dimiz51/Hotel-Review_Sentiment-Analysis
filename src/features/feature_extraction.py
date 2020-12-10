from sklearn.model_selection import train_test_split as split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Transform data for machine learning algorithms
def transform_data(df:pd.DataFrame):
    train, test = split(df,test_size = 0.35,random_state = 45)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train['Review'])
    X_test = vectorizer.transform(test['Review'])
    Y_train = train['Class']
    Y_test = test['Class']
    return X_train, Y_train,X_test,Y_test




if __name__=='__main__':
    root=os.path.dirname(parent_dir_name)
    datapath = os.path.join(root,'data','processed','clean.csv')
    transform_data(pd.read_csv(datapath)) 