from collection import counter
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def print_results(headline, true_value, pred):
    print (headline)
    print ("accuracy: {}").format(accuracy_score(true_value, pred))
    print ("precision: {}").format(precision_score(true_value, pred))
    print ("recall: {}").format(recall_score(true_value, pred))
    print ("f1: {}").format(f1_score(true_value, pred))


data = fetch_datasets()['wine_quality']


X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], random_state=2)

#build normal model
pipeline = make_pipeline(RandomForestClassifier(random_state=42))
model = piepline.fit(X_train, y_test)
prediction = model.predict(X_test)

#build model with SMOTE
smote_pipeline = make_pipeline_imb(SMOTE(random_state=4), RandomForestClassifier(random_state=42))
smote_model = smote_pipeline.fit(X_train, y_train)
smote_prediction = smote_model.predict(X_test)

#build model with under_sampling
nearmiss_pipeline = make_pipeline_imb(NearMiss(random_state=42), RandomForestClassifier(random_state=42))
nearmiss_model = nearmiss_pipeline.fit(X_train, y_train)
nearmiss_prediction = nearmiss_model.predict(X_test)


#print info about both the model_selection
print '\n'
print "Normal Data Distribution: {}\n".format(Counter(data['target']))
X_smote, y_smote = SMOTE().fit_sample(data['data'], data['target'])
print "SMOTE Data Distribution: {}\n".format(Counter(y_smote))
X_nearmiss, y_nearmiss = NearMiss().fit_sample(data['data'], data['target'])
print "Nearmiss Data Distribution: {}\n".format(Counter(y_nearmiss))
