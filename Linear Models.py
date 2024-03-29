#https://scikit-learn.org/stable/supervised_learning.html
import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model

# 1.1. Linear Models
# Ordinary Least Squares
"""
LinearRegression fits a linear model with coefficients  to minimize the residual sum of squares between the observed
targets in the dataset, and the targets predicted by the linear approximation.
"""

# combine all csv file to build training set:
def combine_csv_files():
    path="/home/yavuz/Desktop/ders/super_lig"
    os.chdir(path)
    frame=[]
    b=1
    for files in os.listdir(path):
            print(files)
            data=pd.read_csv(files)
            frame.append(data)
    combined_csv = pd.concat(frame)
    combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')

def linear_model():
    data_train=pd.read_csv("scores.csv") # read data from csv file
    data_x=[] # we gonna select only one feature to build linear regression.
    data_y=[] # this will be our final result.
    for key, value in data_train.iterrows():
        if str(value["HTHG"]) != "nan" and str(value["FTHG"]) != "nan":
            data_x.append(value["HTHG"]) # half-time home gol number
            data_y.append(value["FTHG"]) # full-time home gol number
    data_x=np.array(data_x).reshape(-1, 1)
    data_y=np.array(data_y).reshape(-1, 1)
    """
    [[1.]
     [2.]
     [1.]
    """

    X_train = data_x[:-20] # 80% will our training
    X_test = data_x[-20:] # 20% will be our test
    Y_train = data_y[:-20]
    Y_test = data_y[-20:]


    reg = linear_model.LinearRegression()
    reg.fit(X_train,Y_train)

    y_pred = reg.predict(X_test)
    # The coefficients
    print("Coefficients: \n", reg.coef_)
    # The mean squared error : 0 is perfect prediction
    print("Mean squared error: %.2f" % mean_squared_error(Y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(Y_test, y_pred))


    # Plot outputs
    """plt.scatter(X_test, Y_test, color="black")
    plt.plot(X_test, Y_test, color="blue", linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    
    plt.show()
    """


# 1.1.2. Ridge regression and classification
def Ridge_model():
    from sklearn import linear_model
    data_train = pd.read_csv("scores.csv")  # read data from csv file
    data_x = []  # we gonna select only one feature to build linear regression.
    data_y = []  # this will be our final result.
    for key, value in data_train.iterrows():
        if str(value["HTHG"]) != "nan" and str(value["FTHG"]) != "nan":
            data_x.append(value["HTHG"])  # half-time home gol number
            data_y.append(value["FTHG"])  # full-time home gol number
    data_x = np.array(data_x).reshape(-1, 1)
    data_y = np.array(data_y).reshape(-1, 1)
    """
    [[1.]
     [2.]
     [1.]
    """

    X_train = data_x[:-20]  # 80% will our training
    X_test = data_x[-20:]  # 20% will be our test
    Y_train = data_y[:-20]
    Y_test = data_y[-20:]
    for i in range(1,10):
        reg = linear_model.Ridge(alpha=i/10)
        # reg = linear_model.Lasso(alpha=0.1)
        # LASSO:
        """
        The Lasso is a linear model that estimates sparse coefficients. It is useful in some contexts due to 
        its tendency to prefer solutions with fewer non-zero coefficients, effectively reducing the number 
        of features upon which the given solution is dependent. 
        For high-dimensional datasets with many collinear features, LassoCV is most often preferable. 
        However, LassoLarsCV has the advantage of exploring more relevant values of alpha parameter, 
        and if the number of samples is very small compared to the number of features, it is often faster than LassoCV.

        Alternatively, the estimator LassoLarsIC proposes to use the Akaike information criterion (AIC) and the 
        Bayes Information criterion (BIC). It is a computationally cheaper alternative to find the optimal value of 
        alpha as the regularization path is computed only once instead of k+1 times when using k-fold cross-validation.
        """
        reg.fit(X_train, Y_train)

        y_pred = reg.predict(X_test)
        # The coefficients
        print("Coefficients: \n", reg.coef_)
        # The mean squared error : 0 is perfect prediction
        print(f"Mean squared error {i}  {mean_squared_error(Y_test, y_pred)}")
        # The coefficient of determination: 1 is perfect prediction
        print(f"Coefficient of determination {i}  {r2_score(Y_test, y_pred)}")

#Ridge_model()

# NOTE:
# RidgeCV implements ridge regression with built-in cross-validation of the alpha parameter.
# The object works in the same way as GridSearchCV except that it defaults to Leave-One-Out Cross-Validation:

"""
Akaike information criterion (AIC):
The Akaike information criterion (AIC) is an estimator of prediction error and thereby relative quality of 
statistical models for a given set of data. Given a collection of models for the data, AIC estimates the 
quality of each model, relative to each of the other models. Thus, AIC provides a means for model selection.
AIC is founded on information theory. When a statistical model is used to represent the process that generated the data, 
the representation will almost never be exact; so some information will be lost by using the model to represent the process. 
AIC estimates the relative amount of information lost by a given model: the less information a model loses, the higher 
the quality of that model.
In estimating the amount of information lost by a model, AIC deals with the trade-off between the goodness of fit of the 
model and the simplicity of the model. In other words, AIC deals with both the risk of overfitting and the risk of underfitting.
The Akaike information criterion is named after the Japanese statistician Hirotugu Akaike, who formulated it. 
It now forms the basis of a paradigm for the foundations of statistics and is also widely used for statistical inference.


Bayes Information criterion (BIC):
In statistics, the Bayesian information criterion (BIC) or Schwarz information criterion (also SIC, SBC, SBIC) 
is a criterion for model selection among a finite set of models; models with lower BIC are generally preferred. 
It is based, in part, on the likelihood function and it is closely related to the Akaike information criterion (AIC).

When fitting models, it is possible to increase the likelihood by adding parameters, but doing so may result in overfitting. 
Both BIC and AIC attempt to resolve this problem by introducing a penalty term for the number of parameters in the model; 
the penalty term is larger in BIC than in AIC
"""

# Lower BIC value is the better fitting model.
#view BIC of model
#print(model.bic)

#For example, if Model 1 has an AIC value of 730.5 and Model 2 has an AIC value of 456.3, then Model 2 offers a better fit.
#The absolute values of the AIC are not important.
#print(model.aic)

"""
The MultiTaskLasso is a linear model that estimates sparse coefficients for multiple regression problems jointly: 
y is a 2D array, of shape (n_samples, n_tasks). The constraint is that the selected features are the same for all 
the regression problems, also called tasks.
"""
"""
ElasticNet is a linear regression model trained with both  and -norm regularization of the coefficients. 
This combination allows for learning a sparse model where few of the weights are non-zero like Lasso, 
while still maintaining the regularization properties of Ridge. We control the convex combination of  and  
using the l1_ratio parameter
Elastic-net is useful when there are multiple features that are correlated with one another. 
Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.
"""

def elasticnet():
    from sklearn import linear_model
    data_train = pd.read_csv("scores.csv")  # read data from csv file
    data_x = []  # we gonna select only one feature to build linear regression.
    data_y = []  # this will be our final result.
    for key, value in data_train.iterrows():
        if str(value["HTHG"]) != "nan" and str(value["FTHG"]) != "nan":
            data_x.append(value["HTHG"])  # half-time home gol number
            data_y.append(value["FTHG"])  # full-time home gol number
    data_x = np.array(data_x).reshape(-1, 1)
    data_y = np.array(data_y).reshape(-1, 1)
    """
    [[1.]
     [2.]
     [1.]
    """

    X_train = data_x[:-20]  # 80% will our training
    X_test = data_x[-20:]  # 20% will be our test
    Y_train = data_y[:-20]
    Y_test = data_y[-20:]
    for i in range(1, 10):
        reg = linear_model.MultiTaskElasticNet(alpha=i / 10)
        reg.fit(X_train, Y_train)

        y_pred = reg.predict(X_test)
        # The coefficients
        print("Coefficients: \n", reg.coef_)
        # The mean squared error : 0 is perfect prediction
        print(f"Mean squared error {i}  {mean_squared_error(Y_test, y_pred)}")
        # The coefficient of determination: 1 is perfect prediction
        print(f"Coefficient of determination {i}  {r2_score(Y_test, y_pred)}")
#elasticnet()
"""
class sklearn.linear_model.MultiTaskElasticNet(alpha=1.0, *, l1_ratio=0.5, fit_intercept=True, normalize='deprecated', 
copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, random_state=None, selection='cyclic')"""

"""
LassoLars is a lasso model implemented using the LARS algorithm, and unlike the implementation based on coordinate descent, 
this yields the exact solution, which is piecewise linear as a function of the norm of its coefficients.
"""

#BayesianRidge estimates a probabilistic model of the regression problem as described above. The prior for the coefficient  is given by a spherical Gaussian:
def BayesianRidge_function():
    from sklearn import linear_model
    X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
    Y = [0., 1., 2., 3.]
    reg = linear_model.BayesianRidge()
    reg.fit(X, Y)
    print("Prediction :",reg.predict([[1, 0.]]))
    print("Coefficient: ",reg.coef_)

def logistic_regression_function ():
    """
    Note Feature selection with sparse logistic regression
    A logistic regression with  penalty yields sparse models, and can thus be used to perform feature selection,
    as detailed in L1-based feature selection.

    Note P-value estimation
    It is possible to obtain the p-values and confidence intervals for coefficients in cases of regression
    without penalization. The statsmodels package <https://pypi.org/project/statsmodels/> natively supports this.
    Within sklearn, one could use bootstrapping instead as well.
    """
    from sklearn.linear_model import LogisticRegression
    data_train = pd.read_csv("scores.csv")  # read data from csv file
    data_x = []  # we gonna select only one feature to build linear regression.
    data_y = []  # this will be our final result.
    for key, value in data_train.iterrows():
        if str(value["HTHG"]) != "nan" and str(value["FTHG"]) != "nan":
            data_x.append(value["HTHG"])  # half-time home gol number
            data_y.append(value["FTHG"])  # full-time home gol number
    data_x = np.array(data_x).reshape(-1, 1)
    data_y = np.array(data_y) # do not forget here
    """
    [[1.]
     [2.]
     [1.]
    """

    X_train = data_x[:-20]  # 80% will our training
    X_test = data_x[-20:]  # 20% will be our test
    Y_train = data_y[:-20]
    Y_test = data_y[-20:]
    model = LogisticRegression(solver='liblinear', random_state=0)
    """
    penalty is a string ('l2' by default) that decides whether there is regularization and which approach to use. Other options are 'l1', 'elasticnet', and 'none'.
    dual is a Boolean (False by default) that decides whether to use primal (when False) or dual formulation (when True).
    tol is a floating-point number (0.0001 by default) that defines the tolerance for stopping the procedure.
    C is a positive floating-point number (1.0 by default) that defines the relative strength of regularization. Smaller values indicate stronger regularization.
    fit_intercept is a Boolean (True by default) that decides whether to calculate the intercept 𝑏₀ (when True) or consider it equal to zero (when False).
    intercept_scaling is a floating-point number (1.0 by default) that defines the scaling of the intercept 𝑏₀.
    class_weight is a dictionary, 'balanced', or None (default) that defines the weights related to each class. When None, all classes have the weight one.
    random_state is an integer, an instance of numpy.RandomState, or None (default) that defines what pseudo-random number generator to use.
    solver is a string ('liblinear' by default) that decides what solver to use for fitting the model. Other options are 'newton-cg', 'lbfgs', 'sag', and 'saga'.
    max_iter is an integer (100 by default) that defines the maximum number of iterations by the solver during model fitting.
    multi_class is a string ('ovr' by default) that decides the approach to use for handling multiple classes. Other options are 'multinomial' and 'auto'.
    verbose is a non-negative integer (0 by default) that defines the verbosity for the 'liblinear' and 'lbfgs' solvers.
    warm_start is a Boolean (False by default) that decides whether to reuse the previously obtained solution.
    n_jobs is an integer or None (default) that defines the number of parallel processes to use. None usually means to use one core, while -1 means to use all available cores.
    l1_ratio is either a floating-point number between zero and one or None (default). It defines the relative importance of the L1 part in the elastic-net regularization.
    """
    """
    You should carefully match the solver and regularization method for several reasons:
    'liblinear' solver doesn’t work without regularization.
    'newton-cg', 'sag', 'saga', and 'lbfgs' don’t support L1 regularization.
    'saga' is the only solver that supports elastic-net regularization.
    """

    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)
    # The coefficients
    print("Coefficients: \n", model.coef_)

    from sklearn.metrics import classification_report, confusion_matrix
    # Evaluate the Model
    print("Confusion matrix: \n",confusion_matrix(Y_test.reshape(-1,1),y_pred))
    """
    Example of confusion matrix usage to evaluate the quality of the output of a classifier on the iris data set. 
    The diagonal elements represent the number of points for which the predicted label is equal to the true label, 
    while off-diagonal elements are those that are mislabeled by the classifier. 
    The higher the diagonal values of the confusion matrix the better, indicating many correct predictions.
    """
    print("model score: ",model.score(Y_test.reshape(-1,1),y_pred)) # Y_test should be an array


#logistic_regression_function()

"""
TweedieRegressor implements a generalized linear model for the Tweedie distribution, 
that allows to model any of the above mentioned distributions using the appropriate power parameter. In particular:

power = 0: Normal distribution. Specific estimators such as Ridge, ElasticNet are generally more appropriate in this case.
power = 1: Poisson distribution. PoissonRegressor is exposed for convenience. However, it is strictly equivalent to TweedieRegressor(power=1, link='log').
power = 2: Gamma distribution. GammaRegressor is exposed for convenience. However, it is strictly equivalent to TweedieRegressor(power=2, link='log').
power = 3: Inverse Gaussian distribution.
"""

def SGDClassifier_function():
    """
    Stochastic gradient descent is a simple yet very efficient approach to fit linear models. It is particularly useful
    when the number of samples (and the number of features) is very large. The partial_fit method allows online/out-of-core learning.
    """
    """
    The classes SGDClassifier and SGDRegressor provide functionality to fit linear models for classification and 
    regression using different (convex) loss functions and different penalties. E.g., with loss="log", SGDClassifier 
    fits a logistic regression model, while with loss="hinge" it fits a linear support vector machine (SVM).
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import SGDRegressor
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report, confusion_matrix
    import sklearn
    from sklearn.linear_model import SGDClassifier
    data_train = pd.read_csv("scores.csv")  # read data from csv file
    # you can look colum names using these lines:
    """column_list=[]
    # iterating the columns
    for col in data_train.columns:
        column_list.append(col)
    print(column_list)"""
    selection_of_colums=['FTAG', 'HTHG', 'HTAG', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH',
                         'IWD', 'IWA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD',
                         'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbAH',
                         'BbAHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA']
    X = data_train.loc[:, selection_of_colums]
    y = data_train["FTHG"] # FT home gol number
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=42)
    model = Pipeline([
                    ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
                    ("preprocessing", StandardScaler()),
                    ("classifier", SGDClassifier()),
                ])
    """
    if you faced : ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
    your Y have NaN value.
    """
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    # Evaluate the Model
    print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))


def Perceptron_function():
    """
    Perceptron is a classification algorithm which shares the same underlying implementation with SGDClassifier.
    In fact, Perceptron() is equivalent to SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None).
    The Perceptron is another simple classification algorithm suitable for large scale learning. By default:

    It does not require a learning rate.
    It is not regularized (penalized).
    It updates its model only on mistakes.
    The last characteristic implies that the Perceptron is slightly faster to train than SGD with the hinge loss and
    that the resulting models are sparser.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import SGDRegressor
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report, confusion_matrix
    import sklearn
    from sklearn.linear_model import Perceptron

    data_train = pd.read_csv("scores.csv")  # read data from csv file
    # you can look colum names using these lines:
    """column_list=[]
    # iterating the columns
    for col in data_train.columns:
        column_list.append(col)
    print(column_list)"""
    selection_of_colums=['FTAG', 'HTHG', 'HTAG', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH',
                         'IWD', 'IWA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD',
                         'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbAH',
                         'BbAHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA']
    X = data_train.loc[:, selection_of_colums]
    y = data_train["FTHG"] # FT home gol number
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=42)
    model = Pipeline([
                    ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
                    ("preprocessing", StandardScaler()),
                    ("classifier", Perceptron(tol=1e-3, random_state=0)),
                ])
    """
    if you faced : ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
    your Y have NaN value.
    """
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    # Evaluate the Model
    print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))

#Perceptron_function()

"""
The passive-aggressive algorithms are a family of algorithms for large-scale learning. 
They are similar to the Perceptron in that they do not require a learning rate. However, contrary to the Perceptron, 
they include a regularization parameter C.

For classification, PassiveAggressiveClassifier can be used with loss='hinge' (PA-I) or loss='squared_hinge' (PA-II). 
For regression, PassiveAggressiveRegressor can be used with loss='epsilon_insensitive' (PA-I) or loss='squared_epsilon_insensitive' (PA-II).
"""

# Robustness regression: outliers and modeling errors
"""
An important notion of robust fitting is that of breakdown point: the fraction of data that can be outlying for the fit to start missing the inlying data.

Note that in general, robust fitting in high-dimensional setting (large n_features) is very hard. The robust models here will probably not work in these settings.

Trade-offs: which estimator?

Scikit-learn provides 3 robust regression estimators: RANSAC, Theil Sen and HuberRegressor.

HuberRegressor should be faster than RANSAC and Theil Sen unless the number of samples are very large, i.e n_samples 
>> n_features. This is because RANSAC and Theil Sen fit on smaller subsets of the data. However, both Theil Sen and 
RANSAC are unlikely to be as robust as HuberRegressor for the default parameters.

RANSAC is faster than Theil Sen and scales much better with the number of samples.

RANSAC will deal better with large outliers in the y direction (most common situation).

Theil Sen will cope better with medium-size outliers in the X direction, but this property will disappear in 
high-dimensional settings.

When in doubt, use RANSAC.
"""

# RANSAC: RANdom SAmple Consensus
"""
RANSAC (RANdom SAmple Consensus) fits a model from random subsets of inliers from the complete data set.

RANSAC is a non-deterministic algorithm producing only a reasonable result with a certain probability, 
which is dependent on the number of iterations (see max_trials parameter). It is typically used for linear and 
non-linear regression problems and is especially popular in the field of photogrammetric computer vision.

The algorithm splits the complete input sample data into a set of inliers, which may be subject to noise, and 
outliers, which are e.g. caused by erroneous measurements or invalid hypotheses about the data. The resulting 
model is then estimated only from the determined inliers.

Details of the algorithm
Each iteration performs the following steps:

Select min_samples random samples from the original data and check whether the set of data is valid (see is_data_valid).

Fit a model to the random subset (base_estimator.fit) and check whether the estimated model is valid (see is_model_valid).

Classify all data as inliers or outliers by calculating the residuals to the estimated model (base_estimator.predict(X) - y) 
- all data samples with absolute residuals smaller than or equal to the residual_threshold are considered as inliers.

Save fitted model as best model if number of inlier samples is maximal. In case the current estimated model 
has the same number of inliers, it is only considered as the best model if it has better score.

These steps are performed either a maximum number of times (max_trials) or until one of the special stop criteria 
are met (see stop_n_inliers and stop_score). The final model is estimated using all inlier samples (consensus set) 
of the previously determined best model.

The is_data_valid and is_model_valid functions allow to identify and reject degenerate combinations of random 
sub-samples. If the estimated model is not needed for identifying degenerate cases, is_data_valid should be used 
as it is called prior to fitting the model and thus leading to better computational performance.
"""
#ransac = linear_model.RANSACRegressor()
#ransac.fit(X, y)

"""
The rest of robustness regression are:
Theil-Sen estimator: generalized-median-based estimator
Huber Regression
        The HuberRegressor differs from using SGDRegressor with loss set to huber in the following ways.
        HuberRegressor is scaling invariant. Once epsilon is set, scaling X and y down or up by different values would produce the same robustness to outliers as before. as compared to SGDRegressor where epsilon has to be set again when X and y are scaled.
        HuberRegressor should be more efficient to use on data with small number of samples while SGDRegressor needs a number of passes on the training data to produce the same robustness.
"""


# Quantile Regression
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html#sphx-glr-auto-examples-linear-model-plot-quantile-regression-py

# Polynomial regression: extending linear models with basis functions
# https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions
