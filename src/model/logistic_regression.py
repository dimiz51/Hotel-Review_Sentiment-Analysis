import pandas as pd 
import sys
sys.path.append('./')
from src.features.feature_extraction import transform_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

def logistic_reg(data_file):
    X_train, Y_train,X_test,Y_test = transform_data(pd.read_csv(data_file))
    lr = LogisticRegression(multi_class='ovr')
    lr.fit(X_train,Y_train)
    predictions = lr.predict(X_test)
    print(evaluate_model(predictions,Y_test))

def evaluate_model(predictions,real_classes):
    cm = confusion_matrix(real_classes,predictions,labels=['Very Negative','Negative','Neutral','Positive','Very Positive'])
    print(classification_report(predictions,real_classes))
    return cm

if __name__ == '__main__':
    logistic_reg("./data/processed/clean.csv")
    

