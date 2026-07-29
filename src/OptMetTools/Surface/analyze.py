import numpy as np
from scipy.optimize import curve_fit

class Analyze:
    
    @staticmethod
    def gaussian1d(x, A, mu, sigma, offset):
        """
        Returns a gaussian evaluated at each value in x according to
        the supplied parameters.
        
        Parameters:
            x : np.array(N)
                Supplied input tensor.
            A : float
                Amplitude
            mu : float
                mean
            sigma : float
                variance
            offset : float
                constant offset
        """
        return A*np.exp(-((x-mu)**2/(2*sigma**2)))+offset

    @staticmethod
    def _gaussian_est_p0(predictor, target):
        """
        private method used to estimate initial
        parameters for fit_gaussian_1d.
        """
        A  = np.nanmax(target) * -1
        mu = np.nansum(predictor*target) / np.nansum(target)
        sigma  = np.sqrt( np.nansum(target*((predictor - mu)**2)) / np.nansum(target) )
        offset = -1 * A
        return (A, mu, sigma, offset)

    @staticmethod
    def fit_gaussian_1d(predictor, target):
        """
        Fit's a 1D-Guassian Curve using Scipy Optimize Curvefit.

        Parameters:
            predictor : np.array(N)
                Input tensor (independent variables, etc)
            target : np.array(N)
                Output tensor (dependent variables, etc)


        """
        p0 = Analyze._gaussian_est_p0(predictor, target)
        params, _ = curve_fit(Analyze.gaussian1d, predictor, target, p0=p0, nan_policy='omit')

        return params

    def elliptic_paraboloid(meshgrids, x0, y0, theta, a, b, c):
        r"""
        returns an elliptic paraboloid on inputs X,Y with fitted parameters.

        Parameters:
            meshgrids : np.array(2, nx, ny)
                X and Y inputs, stacked on the first dimension
            x0 : float
                center of paraboloid (X)
            y0 : float
                center of paraboloid (Y)
            theta : float
                Rotation parameter
            a : float
                scaling parameter
            b : foat
                scaling parameter
            c : float
                offset parameter

        Returns:
            np.array(nx,ny)
                surface cosntructed according to input meshgrids & parameters.

        Notes:
            uses the following model

            .. math::
                Z(x,y) = \frac{1}{a^2}(X-x)\cos\theta + (Y-y)\sin\theta)^2 + \frac{1}{b^2}((Y-y)\cos\theta - (X-x)\cos\theta)

        """
        X, Y = meshgrids[0], meshgrids[1]

        u = (X - x0)*np.cos(theta) + (Y - y0)*np.sin(theta)
        v = -(X - x0)*np.sin(theta) + (Y - y0)*np.cos(theta)

        #Z = z0 + k*(u**2) + (1/2)*(s1*(v**2) + s2*u*v)
        Z = (u**2)/(a**2) + (v**2)/(b**2) + c
        return Z


    def fit_elliptic_paraboloid(predictor, target, x0, y0, uncertainty=None):
        """
        Fits an elliptic paraboloid to target data.

        Parameters:
            predictor : np.array(2, nx, ny)
                Supplied input/independent variables.
                predictor[0] is x axis, and predictor[1]
                is y-axis.
            target : np.array(nx,ny) : np.array(nx, ny)
                target/dependent variable to fit to.
            x0 : float
                initial index of center of paraboloid (X)
            y0 : float
                initial index of center of paraboloid (Y)
            uncertainty : float or None
                array of weights in shape of target variable,
                when performing fit.

        Returns:
            coefficients
        """

        if uncertainty is None:
            sigma = np.ones_like(target)
        else:
            #sigma = ((predictor[0] - x0)**2 + (predictor[1] - y0)**2)*0.1 + 0.5
            sigma = uncertainty
        # Make initial guess for parameters:
        p0 = np.array([x0, y0, 0, 1, 1, 0])
        coeffs, _ = curve_fit(
            Analyze.elliptic_paraboloid, predictor, target, p0=p0, nan_policy='omit', sigma=sigma
        )

        return coeffs



