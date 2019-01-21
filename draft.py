import csv
from threading import Thread
from multiprocessing.pool import ThreadPool
from contextlib import contextmanager
import multiprocessing
from multiprocessing import Queue
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from enum import Enum
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import time
import pickle
from sklearn.neural_network import MLPClassifier
fittest=pickle.load(open('fittest.p','rb'))
print(fittest)