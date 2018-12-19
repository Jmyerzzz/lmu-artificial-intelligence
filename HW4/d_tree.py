'''
Sam Chami
Jackson Myers
John Scott
Ben Smith

d_tree.py
'''

import unittest
import numpy as np
import pandas as pd
import scipy
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    """
    The objective of this class is to train a d-tree on preprocessed
    data in a training set, then test its efficacy on a test set.
    """

    def preprocess(self, file_name):
        """
        Method to preprocess data to be in a format that is conducive
        to training.
        """
        names = ("age", "wrk_cls", "edu", "edu_num", "marital_sts",
                 "occu_code", "relation", "race", "sex", "gain",
                 "loss", "hours", "country", "income")

        original_data = pd.read_csv(file_name, dtype=object, names=names, skipinitialspace=True)
        original_data = pd.DataFrame(original_data)

        imp = SimpleImputer(missing_values='?', strategy='most_frequent', verbose=2)
        data = imp.fit_transform(original_data)

        enc = preprocessing.OrdinalEncoder()
        data = enc.fit_transform(data)

        est = preprocessing.KBinsDiscretizer(encode="ordinal")
        data = est.fit_transform(data)

        return data

    def train_model(self, data):
        """
        Trains agent on preprocessed data.
        """
        X, Y = data[:, :-1], data[:, -1]

        classifier = DecisionTreeClassifier()
        classifier.fit(X, Y)
        return classifier

class DecisionTreeTests(unittest.TestCase):
    """Unit Tests"""
    def test(self):
        """Unit Test 1"""
        tree = DecisionTree()
        processed_training_data = tree.preprocess("income-training.csv")
        processed_test_data = tree.preprocess("income-test.csv")

        trained_model = tree.train_model(processed_training_data)
        accuracy = trained_model.score(processed_test_data[:, :-1], processed_test_data[:, -1])
        print("Agent Prediction Accuracy: ", round(accuracy * 100, 2), "%")

if __name__ == "__main__":
    unittest.main()
