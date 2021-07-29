import numpy
import pandas
import torch


class DataPreprocessing:
    def __init__(self, path_to_training_file, path_to_test_file):
        self._path_to_training_file = path_to_training_file
        self._path_to_test_file = path_to_test_file
        self._training_set = None
        self._test_set = None

    def _convert(self, data):
        new_data = []
        for id_users in range(1, self._number_of_users + 1):
            id_movies = data[:, 1][data[:, 0] == id_users]
            id_ratings = data[:, 2][data[:, 0] == id_users]
            ratings = numpy.zeros(self._number_of_movies)
            ratings[id_movies - 1] = id_ratings
            new_data.append(list(ratings))
        new_data = torch.FloatTensor(new_data)
        new_data[new_data == 0.0] = -1.0
        new_data[new_data == 1.0] = 0.0
        new_data[new_data == 2.0] = 0.0
        new_data[new_data >= 3.0] = 1.0
        return new_data

    def get_data(self):
        training_frame = pandas.read_csv(self._path_to_training_file, delimiter="\t")
        training_set = numpy.array(training_frame, dtype="int")
        test_frame = pandas.read_csv(self._path_to_test_file, delimiter="\t")
        test_set = numpy.array(test_frame, dtype="int")
        self._number_of_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
        self._number_of_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))
        training_set = self._convert(training_set)
        test_set = self._convert(test_set)
        return training_set, test_set

    @property
    def number_of_users(self):
        return self._number_of_users

    @property
    def number_of_movies(self):
        return self._number_of_movies
