import numpy as np


class MatrixFactorization:
    def __init__(self, n_factors, learning_rate, regularization):
        """
        Matrix Factorization using Stochastic Gradient Descent (SGD) to factorize a matrix R into two matrices U and V.
        :param n_factors: Number of latent factors
        :param learning_rate: Learning rate
        :param regularization: Regularization parameter
        """
        self._k = n_factors
        self._alpha = learning_rate
        self._lambda = regularization

        self._original_matrix = None

        self._user_matrix = None
        self._item_matrix = None

        self._num_users = None
        self._num_items = None

    def fit(self, matrix, epochs=100) -> np.ndarray:
        """
        Fit the matrix factorization model to the data
        :param matrix: the matrix to factorize
        :param epochs: the number of epochs to train the model
        :return: the predicted ratings
        """
        self._initialize(matrix)

        for i in range(epochs):
            self._sgd()
            loss = self._compute_loss()
            print(f'Epoch {i + 1}/{epochs} - Training Loss: {loss:.4f}')

        return self.predict()

    def predict(self) -> np.ndarray:
        """
        Predict the ratings for every user and item.
        U matrix * V matrix^T
        :return: the predicted ratings
        """
        return self._user_matrix.dot(self._item_matrix.T)

    def _predict(self, i, j) -> np.ndarray:
        """
        Predict the rating of user i for item j
        U[i, :] * V[j, :].T
        :param i: Index of the user
        :param j: Index of the item
        :return: the predicted rating
        """
        return self._user_matrix[i, :].dot(self._item_matrix[j, :].T)

    def _sgd(self):
        """
        Perform Stochastic Gradient Descent to learn the latent factors U and V by minimizing the loss function using the original matrix R.
        :return: None
        """
        for i, j in zip(*self._original_matrix.nonzero()):
            eij = self._original_matrix[i, j] - self._predict(i, j)

            self._user_matrix[i] += self._alpha * (2 * eij * self._item_matrix[j] - self._lambda * self._user_matrix[i])
            self._item_matrix[j] += self._alpha * (2 * eij * self._user_matrix[i] - self._lambda * self._item_matrix[j])

    def _compute_loss(self) -> float:
        """
        Compute the loss (MSE) of the prediction
        :return: the loss of the prediction
        """
        error = 0
        for i, j in zip(*self._original_matrix.nonzero()):
            error = self._original_matrix[i, j] - self._predict(i, j)
            error += error ** 2

        mse = np.mean(error)
        regularization = self._lambda * (np.linalg.norm(self._user_matrix) + np.linalg.norm(self._item_matrix))
        return mse + regularization

    def _initialize(self, matrix):
        self._original_matrix = matrix.copy()
        self._num_users, self._num_items = self._original_matrix.shape

        self._user_matrix = np.random.normal(scale=1. / self._k, size=(self._num_users, self._k))  # U matrix
        self._item_matrix = np.random.normal(scale=1. / self._k, size=(self._num_items, self._k))  # V matrix