# -*- coding: utf-8 -*-
"""random_forest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ykWW6mF6Sjx5y8KyASXMbROxXuEvHkzW
"""

from google.colab import drive
drive.mount('/content/drive')
path = '/content/drive/MyDrive/sem8/siv810/'
import sys
sys.path.append('/content/drive/MyDrive/sem8/siv810/models')
from backend import *

"""Hyperparameters"""

params = []
# different hyperparameters to be tried
criterion = ['gini', 'entropy']
n_estimators = [10, 50, 100]

# merge all hyperparams
params.append(criterion)
params.append(n_estimators)
params = permute_hyperparams(params)

classifier_class = random_forest

"""Dataset 1"""

dataset_id = 1

run_all_for_one_dataset(classifier_class, params, path, dataset_id, run_type = "normal", all = False, normalize = True)

run_all_for_one_dataset(classifier_class, params, path, dataset_id, run_type = "normal", all = False, normalize = False)

run_all_for_one_dataset(classifier_class, params, path, dataset_id, run_type = "pca", all = False, normalize = False)

"""Dataset 2"""

dataset_id = 2

run_all_for_one_dataset(classifier_class, params, path, dataset_id, run_type = "normal", all = False, normalize = True)

run_all_for_one_dataset(classifier_class, params, path, dataset_id, run_type = "normal", all = False, normalize = False)

run_all_for_one_dataset(classifier_class, params, path, dataset_id, run_type = "pca", all = False, normalize = False)

"""Dataset 3"""

dataset_id = 3

run_all_for_one_dataset(classifier_class, params, path, dataset_id, run_type = "normal", all = False, normalize = True)

run_all_for_one_dataset(classifier_class, params, path, dataset_id, run_type = "normal", all = False, normalize = False)

run_all_for_one_dataset(classifier_class, params, path, dataset_id, run_type = "pca", all = False, normalize = False)

"""Dataset 4"""

dataset_id = 4

run_all_for_one_dataset(classifier_class, params, path, dataset_id, run_type = "normal", all = False, normalize = True)

run_all_for_one_dataset(classifier_class, params, path, dataset_id, run_type = "normal", all = False, normalize = False)

run_all_for_one_dataset(classifier_class, params, path, dataset_id, run_type = "pca", all = False, normalize = False)



