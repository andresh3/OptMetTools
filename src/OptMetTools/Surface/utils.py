import numpy as np
from pathlib import Path

class Utils:
    def meshgrid(nx,ny, dtype=None):
        """ Returns a meshgrid with simply the size parameters """
        if dtype:
            X,Y = np.arange(nx,dtype=dtype), np.arange(ny,dtype=dtype)
            return np.meshgrid(X,Y, indexing='ij',)
        else:
            X,Y = np.arange(nx), np.arange(ny)
            return np.meshgrid(X,Y, indexing='ij')

    def circular_mask(nx, ny, cx, cy, r):
        """
        Generates a circular mask of radius `r` about center points (cx, cy) for an
        image of size (nx,ny). 
        """
        X,Y = np.arange(nx), np.arange(ny)
        X,Y = np.meshgrid(X,Y, indexing='ij')

        mask = (X-cx)**2 + (Y-cy)**2 < r**2

        return mask

    def read_ascii(filepath):
        """
        Reads in an ASCII tab-seperated topographic map and
        returns a NumPy meshgrid of the surface.

        Parameters:
            filepath : str
                Absolute or relative path to file

        Returns:
            tuple(np.array(nx,ny), np.array(nx,ny), np.array(nx,ny)
                Returns the X, Y meshgrid axes with the real-valued coordaintes
                as well as the Z array.
        """
        filepath = Path(filepath)

        data = np.loadtxt(filepath, dtype=float)
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        x_vals = np.unique(x)
        y_vals = np.unique(y)

        nx, ny = x_vals.size, y_vals.size

        # Default to nan values where undefined:
        Z = np.full((nx,ny),np.nan)

        # Define a mapping from coordinate to index
        x_index = {val: i for i, val in enumerate(x_vals)}
        y_index = {val: j for j, val in enumerate(y_vals)}

        for xi, yi, zi in data:
            j = y_index[yi]   # row index
            i = x_index[xi]   # column index
            Z[i, j] = zi

        # Shift values for centering
        x_vals -= np.median(x_vals)
        y_vals -= np.median(y_vals)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

        return X,Y,Z

    def regression(Z):
        """
        Performs a regression on all non NaN data points of Z,
        and returns the plane of best fit.
        
        Parameters:
            Z : np.array(nx,ny)
                Input surface

        Returns:
            Plane : np.array(nx,ny)
                Plane of best fit
        """

        # Create meshgrid corresponding to Z and flatten
        # all:
        nx, ny = Z.shape
        X, Y = Utils.meshgrid(nx, ny)
        X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()

        # Filter out NaN values:
        nan_mask = np.isnan(Z)
        X, Y, Z = X[~nan_mask], Y[~nan_mask], Z[~nan_mask]

        # Construct our design matrix:
        # (npx, 3)
        A = np.stack([np.ones_like(X), X, Y],axis=-1)

        coeffs = np.linalg.inv(A.T @ A) @ (A.T @ Z)

        X, Y = Utils.meshgrid(nx, ny)
        plane = coeffs[0] + coeffs[1]*X + coeffs[2]*Y

        return (plane)



        # (X^T x X)^-1 * (X^T)Y


