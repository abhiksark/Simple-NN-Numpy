from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

digits = load_digits()


x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

class NN(object):
    def __init__(self):
        pass
    def fit(self, x_train,y_train):
        pass