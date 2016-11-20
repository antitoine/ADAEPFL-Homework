#!/usr/lib/python3

import numpy as np
import pandas as pd

from dateutil import relativedelta
from datetime import date
import itertools
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


def compute_age(row):
    """
        Given a player, function returns the years of player.
        row: Row of the DataFrame, representing a dyad which contains a player.
    """

    data_date = date(2013, 1, 1)
    delta = relativedelta.relativedelta(data_date, row['birthday'])

    return delta.years


def regulate_number_of_cards(row, cards_name, k=1):
    """
    Given a player, function analyzes the skin color and balance the number of received cards if player is black.

    row: Row of the DataFrame, representing a dyad which contains a player.
    cardsName: Type of received cards for the player
    k: factor of impact (more k is bigger more the regulation is important)
    """

    nb_cards = row[cards_name]

    if row['associationScore'] > 0:
        coef = (row['rater'] / 100) * row['associationScore']
    elif row['associationScore'] < 0:
        coef = (1 - (row['rater'] / 100)) * row['associationScore']
    else:
        coef = 0

    nb_cards += nb_cards * coef * k

    return nb_cards


def compute_skin_separation_percentage(estimation, real_classes):
    """
    This function computes the the separation of players (in different clusters) regarding to their skin color.

    estimation: Represents in which cluster are marked the players
    real_classes: Represents the skin color of all the players
    """

    if len(estimation) != len(real_classes):
        print('The size of the two arrays are not equals')
        return 0
    
    size = len(estimation)
    equal = 0
    for i in range(0, size):
        # 
        if estimation[i] == real_classes[i]:
            equal += 1
    different = size - equal
    return abs(equal - different) / size


def get_random_forests(nb_trees=[10], min_leaf=[2], min_split=[2]):
    """
    Given a set of parameters, function create a set of RandomForestClassifier for each unique set of parameters.

    These parameters refer to parameters of RandomForestClassifier's constructor.
    For more detail, please visit: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    rfs = []

    for tree_param in nb_trees:
        for min_leaf_param in min_leaf:
            for min_split_param in min_split:
                rfs.append(RandomForestClassifier(n_jobs=-1, n_estimators=tree_param, min_samples_leaf=min_leaf_param, min_samples_split=min_split_param))
    return rfs


def run_once(classifiers, data, features, classes):
    """
    For each classifier, we run fit/predict operations of the aforesaid classifier.

    classifiers: set of classifiers to be used
    data: Data to be used for train and test operations
    features: Features to be considered
    classes: Classes to be predicted
    """

    # We keep only interesting features
    x = data[features]
    # We then use LabelEncoder to transform classes, which are denoted as string by default
    y = preprocessing.LabelEncoder().fit_transform(data[classes])
    # Train/Test/Split to obtain randomly-generated sets for train and test parts
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)
    # We finally binarize the obtained labels to compute ROC AUC later
    lb = preprocessing.LabelBinarizer()
    y_test_binary = lb.fit_transform(y_test)

    results = []

    for classifier in classifiers:
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        y_probabilities = classifier.predict_proba(x_test)

        # We store important data relative to the classifier and its results
        result = {
            'classifier': classifier,
            'scores': {
                'accuracy': metrics.accuracy_score(y_test, predictions),
                'f-score': metrics.f1_score(y_test, predictions, average='binary'),
                'roc': metrics.roc_curve(y_test_binary, y_probabilities[:, 1]),
                'roc_auc': metrics.roc_auc_score(y_test_binary, y_probabilities[:, 1])
            },
            'confusion_matrix': metrics.confusion_matrix(y_test, predictions)
        }

        results.append(result)

    return results


def run_cross_validation(classifiers, data, features, classes):
    """
    For each given classifier, function run cross validation to obtain mean accuracy

    classifiers: Classifiers to be considered for cross validation
    data: Data to be used with classifiers (train and test)
    features: Relevant features to be considered
    classes: Classes to be predicted
    """

    results = []
    # We first transform classes to obtain binary values
    y = preprocessing.LabelEncoder().fit_transform(data[classes])
    y_binary = preprocessing.LabelBinarizer().fit_transform(y)

    for classifier in classifiers:
        accuracy = cross_val_score(classifier, data[features], data[classes], cv=10, scoring='accuracy')
        f1 = cross_val_score(classifier, data[features], data[classes], cv=10, scoring='f1_macro')
        #roc_auc = cross_val_score(classifier, data[features], y_binary, cv=10, scoring='roc_auc')

        # We store important data relative to classifier and cross validation scores
        result = {
            'classifier': classifier,
            'scores': {
                'accuracy_mean': np.mean(accuracy),
                'f1_mean': np.mean(f1)
                #'roc_auc_mean': np.mean(roc_auc)
            }
        }

        results.append(result)

    return results


def retrieve_above_threshold(classifier, features, threshold=10):
    """
    Given a classifier, function return features for which importance (relative to the most important feature) is above threshold

    classifier: Classifier to be considered
    features: Features used with classifier
    threshold: Threshold used to discriminate features
    """

    feature_importance = classifier.feature_importances_
    # We compute importance score relative to the most important feature used by the classifier
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    # We obtain indexes of all features which are the most important (importance score abode threshold)
    index = np.where(feature_importance > threshold)[0]

    return pd.Series(data=[feature_importance[i] for i in index], index=[features[i] for i in index])


def get_mean_squared_error(classifier, data, features, classes, samples, nbCrossVal):
    """
    Determines cross-validated training and test mean_square_error for different training set sizes.
    A cross-validation generator splits the whole dataset k times in training and test data.
    Subsets of the training set with varying sizes will be used to train the estimator and a mean
    square error for each training subset size and the test set will be computed
    ----------
    classifier : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    data : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    features : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.   
    nb_samples : number of slice of data for the graph
    nbCrossVal : number of cross-validation, more is higher better become the prediciton.
    """
    error_with_cv_list = []
    error_without_cv_list = []
    
    for sampleNumber in samples:

        temp = data.copy()
        
        # take a ramdom sample of size sampleNumber
        sampleData = temp.sample(n=sampleNumber,replace=True)
    
        # select feature.
        x = sampleData[features]
        y = preprocessing.LabelEncoder().fit_transform(sampleData[classes])

        # train without
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)
        y_test_binary = preprocessing.LabelBinarizer().fit_transform(y_test)

        #prediction of our training part
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)

        #training part.
        error_with_cv = cross_val_score(classifier, x, y, cv=nbCrossVal,
                                        scoring='neg_mean_squared_error')
        
        error_without_cv = mean_squared_error(y_test, predictions, multioutput='raw_values')
        
        error_with_cv_list.append(abs(error_with_cv))
        error_without_cv_list.append(error_without_cv)
        
    return (samples , error_with_cv_list , error_without_cv_list)

def learning_curve_mean_squared_error (classifier, data, features, classes, nb_samples,nbCrossVal,ylim = None):
    """
    Generate a simple plot of the test and training learning curve depending of RMSE.
    Parameters
    ----------
    classifier : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    data : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    features : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.   
    nb_samples : number of slice of data for the graph
    nbCrossVal : number of cross-validation, more is higher better become the prediciton.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    """
    plt.figure()
    plt.title('Difference cross-validation, training with RMSE')
    ylim = None
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("RMSE")

    # select the different sample
    x = math.floor(len(data)/nb_samples)
    value_sample = []
    while x < len(data):
        value_sample.append(x)
        x += x
    value_sample.append(len(data))

    # return the 
    train_sizes, train_scores, test_scores = get_mean_squared_error(
                      classifier,data, features,classes,value_sample,nbCrossVal)

    # We take the best result by taking the minimum.
    cv_mean_squared_error = np.min(train_scores, axis=1)
    mean_squared_error  = np.min(test_scores, axis=1)

    # plotting the result
    plt.grid()
    plt.plot(train_sizes, mean_squared_error, 'o-', color="r",
             label="Training mean squared root")
    plt.plot(train_sizes, cv_mean_squared_error, 'o-', color="g",
             label="Cross-validation mean squared root")
    plt.legend(loc="best")
    return plt


# Function is defined in sklearn documentation and was slightly modified to fit with our needs and the situation
# See: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Function is defined in sklearn documentation
# See: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, data, features, classes, ylim=None):
    """
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    data : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    features : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    /* INTERNAL VARIABLE ONLY HERE */
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    """
    X = data[features]
    y = preprocessing.LabelEncoder().fit_transform(data[classes])

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt