import numpy as np

class AdalineGD:
    """
    Adaptive Linear Neuron (Adaline) classifier using batch gradient descent.

    Parameters
    ----------
    lr : float
        Learning rate (between 0.0 and 1.0).
    n_iter : int
        Number of passes (epochs) over the training dataset.
    random_state : int
        Random seed for initializing weights.

    Attributes
    ----------
    w_ : 1d-array
        Weights after training.
    b_ : float
        Bias unit after training.
    losses_ : list
        Mean squared error loss in each epoch.
    """

    def __init__(self, lr=0.01, n_iter=100, random_state=1):
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data using batch gradient descent.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """
        # Initialize weights with small random numbers
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0
        self.losses_ = []

        # Training loop
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output

            # Gradient descent weight updates
            self.w_ += self.lr * 2.0 * X.T @ errors / X.shape[0]
            self.b_ += self.lr * 2.0 * errors.mean()

            # Compute mean squared error
            loss = (errors**2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        """
        Compute net input (weighted sum + bias).
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        
        Returns
        -------
        float or ndarray
            Net input
        """
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """
        Compute linear activation (identity function).

        Parameters
        ----------
        X : array-like
            Net input
        
        Returns
        -------
        X : array-like
            Activated output (unchanged)
        """
        return X

    def predict(self, X):
        """
        Return class label after unit step.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        
        Returns
        -------
        class_labels : array, shape = [n_samples]
            Predicted class labels (0 or 1)
        """
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
