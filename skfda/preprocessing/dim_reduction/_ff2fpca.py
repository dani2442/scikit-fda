"""
Functional to Functional Principal Component Analysis
====================================================

Explores the way to do functional to functional principal component analysis.
"""

# Author: Daniel López-Montero
# License: MIT

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, TypeVar

import skfda
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import FDataBasis
from skfda.misc.regularization import L2Regularization
from skfda.representation import FData
from skfda.typing._numpy import ArrayLike, NDArrayFloat

Function = TypeVar("Function", bound=FData)
WeightsCallable = Callable[[np.ndarray], np.ndarray]

class FF2FPCA:
    r"""
    Functional to Functional PCA.

    Class that implements functional to functional PCA for grid representations of the data. 
    """

    def __init__(
        self,
        n_components: int | None = 1,
        centering: bool = True,
        regularization: L2Regularization[FData] | None = None,
        _weights: ArrayLike | WeightsCallable | None = None,
        _mean: ArrayLike | WeightsCallable | None = None,
    ) -> None:
        self.n_components = n_components
        self.centering = centering
        self.regularization = regularization
        self._weights = _weights
        self._mean = _mean

    def _center_if_necessary(
        self,
        X: Function,
        *,
        learn_mean: bool = True,
    ) -> Function:

        if learn_mean:
            
            self._mean = X.mean()

        return X - self._mean if self.centering else X

    def _fit_grid(
        self,
        X: FDataGrid,
        y: object = None
    ) -> FF2FPCA:
        """
        Compute the first n_components principal components and saves them.

        Args:
            X: The functional data object to be analyzed.
            y: Ignored.

        Returns:
            self

        """
        # data matrix initialization
        n, N, p = X.data_matrix.shape

        X = X.copy()
        X.data_matrix = np.fft.fft(X.data_matrix, axis=1)

        # if centering is True then subtract the mean function to each function
        # in FDataBasis
        X = self._center_if_necessary(X)

        cov = (X.data_matrix.transpose(1, 2, 0) @ X.data_matrix.transpose(1,0,2)) / (n - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Order by descending eigenvalue
        idx = np.argsort(eigenvalues, axis=1)[:, ::-1]
        eigenvalues = np.take_along_axis(eigenvalues, idx, axis=-1)
        eigenvectors = np.take_along_axis(eigenvectors, idx[:, np.newaxis, :], axis=-1)

        # Align eigenvector signs
        for i in range(1, eigenvectors.shape[0]):
            eigenvectors[i] *= np.sign(np.sum(eigenvectors[i] * eigenvectors[i - 1], axis=0, keepdims=True))
        
        return FF2FPCA(self.n_components, _mean = self._mean, _weights = eigenvectors)

    def _transform_grid(
        self,
        X: FDataGrid,
        y: object = None,
    ) -> NDArrayFloat:
        """
        Compute the ``n_components`` first principal components score.

        Args:
            X: The functional data object to be analysed.
            y: Ignored.

        Returns:
            Principal component scores.

        """
        # in this case its the coefficient matrix multiplied by the principal
        # components as column vectors

        return (  # type: ignore[no-any-return]
            X.data_matrix.transpose(1,0,2) @ self._weights[:, :, :self.n_components]
        )

    def fit(
        self,
        X: FData,
        y: object = None,
    ) -> FF2FPCA:
        """
        Compute the n_components first principal components and saves them.

        Args:
            X: The functional data object to be analysed.
            y: Ignored.

        Returns:
            self

        """
        if isinstance(X, FDataGrid):
            return self._fit_grid(X, y)
        elif isinstance(X, FDataBasis):
            raise Exception("FDataBasis is not implemented yet.")
            return self._fit_basis(X, y)

        raise AttributeError("X must be either FDataGrid or FDataBasis")

    def transform(
        self,
        X: FData,
        y: object = None,
    ) -> NDArrayFloat:
        """
        Compute the ``n_components`` first principal components scores.

        Args:
            X: The functional data object to be analysed.
            y: Only present because of fit function convention

        Returns:
            Principal component scores.

        """
        X = X.copy()
        X.data_matrix = np.fft.fft(X.data_matrix, axis=1)

        X = self._center_if_necessary(X, learn_mean=False)

        if isinstance(X, FDataGrid):
            return self._transform_grid(X, y)
        elif isinstance(X, FDataBasis):
            raise Exception("Unfortunately, F2FPCA does not admit basis yet (use grid instead).")
            return self._transform_basis(X, y)

        raise AttributeError("X must be either FDataGrid or FDataBasis")

    def fit_transform(
        self,
        X: FData,
        y: object = None,
    ) -> NDArrayFloat:
        """
        Compute the n_components first principal components and their scores.

        Args:
            X: The functional data object to be analysed.
            y: Ignored

        Returns:
            Principal component scores.

        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(
        self,
        pc_scores: NDArrayFloat,
    ) -> FData:
        """
        Compute the recovery from the fitted principal components scores.

        In other words,
        it maps ``pc_scores``, from the fitted functional PCs' space,
        back to the input functional space.
        ``pc_scores`` might be an array returned by ``transform`` method.

        Args:
            pc_scores: ndarray (n_samples, n_components).

        Returns:
            A FData object.

        """
        # Calculate the reconstruction
        result = pc_scores @  self._weights[:, :, :self.n_components].transpose(0, 2, 1)

        result = result.transpose(1,0,2) + self._mean.data_matrix
        result = np.fft.ifft(result, axis=1).real

        r = self._mean.copy(
                data_matrix=result,
                sample_names=(None,) * pc_scores.shape[1]
            )
        return r


