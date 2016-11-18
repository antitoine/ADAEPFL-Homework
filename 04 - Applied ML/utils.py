#!/usr/lib/python3

import numpy as np

from dateutil import relativedelta
from datetime import date
import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

import matplotlib.pyplot as plt


def compute_age(row):
    """
        Given a player, function returns the years of player.
        row: Row of the DataFrame, representing a dyad which contains a player.
    """
    data_date = date(2013, 1, 1)
    delta = relativedelta.relativedelta(data_date, row['birthday'])

    return delta.years


def pondered_number_of_cards(row, cards_name):
    """
    Given a player, function analyzes the skin color and balance the number of received cards if player is black.

    row: Row of the DataFrame, representing a dyad which contains a player.
    cardsName: Type of received cards for the player
    """
    nb_cards = row[cards_name]

    if row['associationScore'] > 0:
        coef = (row['rater'] / 100) * row['associationScore']
    elif row['associationScore'] < 0:
        coef = (1 - (row['rater'] / 100)) * row['associationScore']
    else:
        coef = 0

    nb_cards += nb_cards * coef

    return nb_cards


def get_random_forests(nb_trees=[10], min_leaf=[2], min_split=[2]):
    rfs = []
    for tree_param in nb_trees:
        for min_leaf_param in min_leaf:
            for min_split_param in min_split:
                rfs.append(RandomForestClassifier(n_jobs=-1, n_estimators=tree_param, min_samples_leaf=min_leaf_param, min_samples_split=min_split_param))
    return rfs


def run_once(classifiers, data, features, classes):
    x = data[features]
    y = preprocessing.LabelEncoder().fit_transform(data[classes])
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)
    y_test_binary = preprocessing.LabelBinarizer().fit_transform(y_test)

    results = []

    for classifier in classifiers:
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        y_probabilities = classifier.predict_proba(x_test)

        result = {
            'classifier': classifier,
            'scores': {
                'accuracy': metrics.accuracy_score(y_test, predictions),
                'f-score': metrics.f1_score(y_test, predictions, average='macro'),
                #'roc_auc': metrics.roc_auc_score(y_test_binary, y_probabilities)
            },
            'confusion_matrix': metrics.confusion_matrix(y_test, predictions)
        }

        results.append(result)

    return results


def run_cross_validation(classifiers, data, features, classes):

    results = []
    y_binary = preprocessing.LabelBinarizer().fit_transform(data[classes])

    for classifier in classifiers:
        accuracy = cross_val_score(classifier, data[features], data[classes], cv=10, scoring='accuracy')
        roc_auc = cross_val_score(classifier, data[features], y_binary, cv=10, scoring='roc_auc')

        result = {
            'classifier': classifier,
            'scores': {
                'accuracy_mean': np.mean(accuracy),
                #'roc_auc_mean': np.mean(roc_auc)
            }
        }

        results.append(result)

    return results


# Function is defined in sklearn documentation and was slightly modified to fit with our needs and our situation
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

