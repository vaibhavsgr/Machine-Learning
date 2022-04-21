from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import matplotlib.pylot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, train_test_split
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbours
from imblearn.under_sampling import EditedNearestNeighbours, RepeatedNearestNeighbours, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalanceCascade, EasyEnsemble
from sklearn.ensemble import AdaBoostClassifier
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
from itable import PrettyTable, TableStyle, CellStyle
from pylab import *
%pylab inline
pylab.rcParams['figure.figsize'] = (12,6)
plt.style.use('fivethirtyeight')

#Generate data with two classes
X, y = make_classification(class_sep=1.2, weights=[0.1, 0.9], n_informative=3,
                            n_redundant=1, n_features=5, n_clusters_per_class=1,
                            n_samples=10000, flip_y=0, random_state=10)

pca = PCA(n_components=2)
X = pca.fit_transform(X)

y = y.astype('str')
y[y=='1'] = 'L'
y[y=='0'] = 'S'

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

X_1, X_2 = X_train[y_train=='S'], X_train[y_train=='L']


#Scatter plot of the dataset
plt.scatter(zip(*X_1)[0], zip(*X_1)[1], color='#labc9c')
plt.scatter(zip(*X_2)[0], zip(*X_2)[1], color='#e67e22')


x_coords = zip(*X_1)[0] + zip(*X_2)[0]
y_coords = zip(*X_1)[1] + zip(*X_2)[1]
plt.axis([min(x_coords), max(x_coords), min(y_coords, max(y_coords)])

plt.title("Original Dataset")
plt.show()
