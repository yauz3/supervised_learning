import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
import sklearn
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.utils import shuffle
import numpy as np
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection
import os
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
import pickle
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
import statsmodels.api as sm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import BallTree
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVC
from sklearn.svm import OneClassSVM
from sklearn.svm import l1_min_c
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
import statsmodels.api as sm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVC
from sklearn.svm import OneClassSVM
from sklearn.svm import l1_min_c
from sklearn import tree
import csv
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from random import randint
from tpot.builtins import ZeroCount
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import math
import random
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.preprocessing import FunctionTransformer
from copy import copy
from sklearn.preprocessing import RobustScaler


#training
filename="train_with_gap.csv"
data = pd.read_csv("%s"%filename, index_col=0)
selection_list = ["vina_rmsd_ub"]

X = data.loc[:, selection_list]
y = data["True or False"]

#test
test_filename="test.csv"
data_t = pd.read_csv("%s"%test_filename, index_col=0)

data_test = data_t.loc[:, selection_list]
data_y_test = data_t["True or False"]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,test_size=0.25, random_state=42)


for max in range(0,1):
    option_1 = KNeighborsClassifier()
    option_2 = DecisionTreeClassifier()
    option_3 = RandomForestClassifier(class_weight="balanced_subsample")
    option_31 = RandomForestClassifier(n_estimators=270,max_depth=14,min_samples_split=16,min_samples_leaf=4,min_weight_fraction_leaf=1/2350,
                                       class_weight="balanced_subsample")
    option_4 = AdaBoostClassifier() # olabilir tekrarlı fakat
    option_5 = HistGradientBoostingClassifier() # 71-71-81 tekrarlı
    option_6 = GradientBoostingClassifier()
    option_7 = ExtraTreesClassifier(class_weight="balanced_subsample")
    option_8 = BaggingClassifier()
    option_9 = svm.SVC(probability=True,cache_size=25000)
    option_10 = RidgeClassifier()
    option_11 = RidgeClassifierCV()
    option_12 = SGDClassifier(loss="perceptron")
    option_13 = BernoulliNB()
    option_14 = NearestCentroid()
    option_15 = MLPClassifier()
    option_16 = LinearSVC()
    option_17 = OneClassSVM()
    option_18 = GaussianNB()
    option_19 = DummyClassifier()
    option_20 = BaggingClassifier()
    option_21 = XGBClassifier()
    option_22 = NearestNeighbors()
    option_23 = ExtraTreesClassifier(max_features=1.0, min_samples_split=16)
    clf2 = ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                                min_samples_leaf=1, max_features='log2',
                                max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight="balanced",
                                ccp_alpha=0.0, max_samples=None)
    mail=ExtraTreesClassifier(n_estimators=200,max_depth=14,min_samples_split=16,min_samples_leaf=4,min_weight_fraction_leaf=1/2350)

    #####################################################################################################################
    #####################################################################################################################

    layer=[("bes", option_5),("opt", option_6),("optt", option_7) ]
    voiting=VotingClassifier(estimators=layer, voting="soft")
    stacking=StackingClassifier(estimators=layer, final_estimator=mail)
    model_final = Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("preprocessing", StandardScaler()),
        ("classifier", option_9),
    ])
    """exported_pipeline = make_pipeline(
        make_union(
            FunctionTransformer(copy),
            FunctionTransformer(copy)
        ),
        option_18,
    )
    model_final = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("preprocessing", RobustScaler()),
        ("classifier", exported_pipeline),
    ])"""
    model = model_final.fit(X_train, y_train)
    """
    StandardScaler
    RobustScaler
    strategies = ['mean', 'median', 'most_frequent', 'constant']"""
    print("\nspileted test")
    y_pred = model.predict(X_test)
    acc_2 = metrics.accuracy_score(y_test, y_pred)
    print("accucary %s" % acc_2)
    y_trained = model.predict(X_train)
    accy_3 = metrics.accuracy_score(y_trained, y_train)
    print("acc train: %s" % accy_3)
    # print("\nhere %s" % ((acc + acc_2) / 2))
    matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print('unseen F1 score: %f' % f1)
    print("matrix\n", matrix)
    # cross-validation
    """print("cross-validation")
    print(cross_val_score(model_final, X, y, cv=10))"""

    # save the model to disk
    filename = 'svm.sav'
    pickle.dump(model, open(filename, 'wb'))

    """accucary_coach = prediction_bindingdb_david(bindingdb,"model_no_fpocket.sav")
    if accucary_coach >= 72:
        filename = 'model_%s.sav' % (str(max) + "_" + str(min))
        pickle.dump(model, open(filename, 'wb'))
    output_dictionary[max, min] = accucary_coach
print(output_dictionary)"""




