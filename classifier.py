from time import time
from functools import partial

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from detection import extract_features
from util import log, pickle, unpickle

from config import classifier


class Classifier:
    def __init__(self, parameters=classifier):
        self.parameters = parameters
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.x_scaler = None

    def create_features(self, save=True):
        param = self.parameters
        vehicles = param['vehicles']
        non_vehicles = param['non_vehicles']
        log('Vehicles     : {}'.format(len(vehicles)))
        log('Non-vehicles : {}'.format(len(non_vehicles)))

        orient, pix_per_cell, cell_per_block = param['orient'], param['pix_per_cell'], param['cell_per_block']
        extract_features_partial = partial(extract_features,
                                           color_space=param['color_space'],
                                           spatial_size=param['spatial_size'],
                                           hist_bins=param['hist_bins'],
                                           orient=param['orient'],
                                           pix_per_cell=param['pix_per_cell'],
                                           cell_per_block=param['cell_per_block'],
                                           hog_channel=param['hog_channel'],
                                           spatial_feat=param['spatial_feat'],
                                           hist_feat=param['hist_feat'],
                                           hog_feat=param['hog_feat'])

        vehicles_features = extract_features_partial(vehicles)
        non_vehicles_features = extract_features_partial(non_vehicles)

        x = np.vstack((vehicles_features, non_vehicles_features)).astype(np.float64)
        self.x_scaler = StandardScaler().fit(x)
        x_scaled = self.x_scaler.transform(x)

        # Define the labels vector
        y = np.hstack((np.ones(len(vehicles)), np.zeros(len(non_vehicles))))

        # Split up data into randomized training and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_scaled, y,
                                                                                test_size=param['test_size'])

        log(
            'Using {} orientations {} pixels per cell {} cells per block'.format(orient, pix_per_cell, cell_per_block))
        log('Feature vector length: {}'.format(self.x_train.shape[0]))

        if save:
            log('Pickling features')
            pickle(
                {'x_train': self.x_train, 'x_test': self.x_test, 'y_train': self.y_train, 'y_test': self.y_test,
                 'x_scaler': self.x_scaler, 'parameters': param},
                param["pickle_features"]
            )

    def load_features(self):
        features = unpickle(self.parameters["pickle_features"])
        self.x_train = features['x_train']
        self.x_test = features['x_test']
        self.y_train = features['y_train']
        self.y_test = features['y_test']
        self.x_scaler = features['x_scaler']
        self.parameters = features['parameters']

    def create_model(self, model_class=LinearSVC, save=True):
        self.model = model_class()
        start = time()
        self.model.fit(self.x_train, self.y_train)
        log('Training time : {:.0f}s'.format(time() - start))
        log('Test accuracy : {:.2f}%'.format(100 * self.model.score(self.x_test, self.y_test)))

        if save:
            log('Pickling model')
            pickle(self.model, self.parameters["pickle_model"])

    def load_model(self):
        if self.x_test is None:
            self.load_features()
        self.model = unpickle(self.parameters["pickle_model"])
