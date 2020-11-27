from sklearn.model_selection import train_test_split as split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Transform data for machine learning algorithms
def transform_data(df:pd.DataFrame):
    train, test = split(df,test_size = 0.25,random_state = 45)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train['Review'])
    X_test = vectorizer.transform(test['Review'])
    Y_train = train[['Rating','Class']]
    Y_test = test[['Rating','Class']]
    return X_train, Y_train,X_test,Y_test




if __name__=='__main__':
    transform_data(pd.read_csv('./data/processed/clean.csv')) 