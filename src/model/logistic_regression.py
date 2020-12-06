import pandas as pd 
import sys
sys.path.append('./')
from src.features.feature_extraction import transform_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle



# Loads tf-idf transformed data and applies the logistic regression classification algorithm on them
""" 
As we are facing a multiclass(5 sentiment polarity based classes) problems
we are gonna use OvR(One-Versus-Rest) tecnique for our logistic regression algorithm,
meaning we are gonna transform the problem into 5 separate propability estimation problems.
Also, as our dataset has huge sentiment class imbalances, meaning the positive sentiment samples 
outnumber by a lot the neutral and negative ones, our algorithm will propably bring better results 
by using class weights during the training phase.
"""


def logistic_reg(data_file):
    X_train, Y_train,X_test,Y_test = transform_data(pd.read_csv(data_file))
    lr = LogisticRegression(multi_class='ovr',class_weight='balanced')
    lr.fit(X_train,Y_train)
    predictions = lr.predict(X_test)
    evaluate_model(predictions,Y_test)
    # save the model to disk
    filename = './models/logistic_reg.sav'
    pickle.dump(lr, open(filename, 'wb'))
 

# Evaluate model performance,print classification report, plot and save confusion matrix as a seaborn heatmap
def evaluate_model(predictions,real_classes):
    class_labels = ['Very Negative','Negative','Neutral','Positive','Very Positive']
    cm = confusion_matrix(real_classes,predictions,labels=class_labels)
    cm_df = pd.DataFrame(cm,index = class_labels,columns = class_labels)
    plot_confmatrix(cm_df)
    print(classification_report(predictions,real_classes))
    
       
# Confusion matrix plotting function
def plot_confmatrix(confusion_matrix):
    fig = plt.figure(figsize=(12,10))
    plt.title('Logistic Regression Model Confusion Matrix',fontsize=18)
    heatmap = sns.heatmap(confusion_matrix,annot= True, annot_kws={"size": 16},cmap='Blues',fmt='d') 
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
    fig.savefig('./data/plots/logistic_reg.png')

if __name__ == '__main__':
    logistic_reg("./data/processed/clean.csv")
    

