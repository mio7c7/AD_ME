import numpy as np
from scipy import stats
from numpy.linalg import inv

class Distribution:
    def reset_params(self):
        raise NotImplementedError()

    def pdf(self, x):
        raise NotImplementedError()

    def update_params(self, x):
        raise NotImplementedError()

class MultivariateT(Distribution):
    def __init__(
        self,
        dims: int = 1,
        dof: int = 0,
        kappa: int = 1,
        mu: float = -1,
        scale: float = -1,
    ):
        """
        Create a new predictor using the multivariate student T distribution as the posterior predictive.
            This implies a multivariate Gaussian distribution on the data, a Wishart prior on the precision,
             and a Gaussian prior on the mean.
             Implementation based on Haines, T.S., Gaussian Conjugate Prior Cheat Sheet.
        :param dof: The degrees of freedom on the prior distribution of the precision (inverse covariance)
        :param kappa: The number of observations we've already seen
        :param mu: The mean of the prior distribution on the mean
        :param scale: The mean of the prior distribution on the precision
        :param dims: The number of variables
        """
        # We default to the minimum possible degrees of freedom, which is 1 greater than the dimensionality
        if dof == 0:
            dof = dims + 1
        # The default mean is all 0s
        if mu == -1:
            mu = [0] * dims
        else:
            mu = [mu] * dims

        # The default covariance is the identity matrix. The scale is the inverse of that, which is also the identity
        if scale == -1:
            scale = np.identity(dims)
        else:
            scale = np.identity(scale)

        # The dimensionality of the dataset (number of variables)
        self.dims = dims

        # Each parameter is a vector of size 1 x t, where t is time. Therefore each vector grows with each update.
        self.dof = np.array([dof])
        self.kappa = np.array([kappa])
        self.mu = np.array([mu])
        self.scale = np.array([scale])

        self.dofT = self.dof.copy()
        self.kappaT = self.kappa.copy()
        self.muT = self.mu.copy()
        self.scaleT = self.scale.copy()

    def reset_params(self):
        self.dofT = self.dof.copy()
        self.kappaT = self.kappa.copy()
        self.muT = self.mu.copy()
        self.scaleT = self.scale.copy()

    def pdf(self, data: np.array):
        """
        Returns the probability of the observed data under the current and historical parameters
        Parmeters:
            data - the datapoints to be evaualted (shape: 1 x D vector)
        """
        t_dof = self.dof - self.dims + 1
        expanded = np.expand_dims((self.kappa * t_dof) / (self.kappa + 1), (1, 2))
        loc = self.mu[-1]
        shape = inv(expanded * self.scale)[-1]
        try:
            ret = stats.multivariate_t.pdf(
                x=data,
                loc=loc,
                df=t_dof[-1],
                shape=shape)
        except:
            pass
        return ret

    def update_params(self, data: np.array, **kwargs):
        """
        Performs a bayesian update on the prior parameters, given data
        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        centered = data - self.mu

        # We simultaneously update each parameter in the vector, because following figure 1c of the BOCD paper, each
        # parameter for a given t, r is derived from the same parameter for t-1, r-1
        # Then, we add the prior back in as the first element
        self.scale = np.concatenate(
            [
                self.scale[:1],
                inv(
                    inv(self.scale)
                    + np.expand_dims(self.kappa / (self.kappa + 1), (1, 2))
                    * (np.expand_dims(centered, 2) @ np.expand_dims(centered, 1))
                ),
            ]
        )
        self.mu = np.concatenate(
            [
                self.mu[:1],
                (np.expand_dims(self.kappa, 1) * self.mu + data)
                / np.expand_dims(self.kappa + 1, 1),
            ]
        )
        self.dof = np.concatenate([self.dof[:1], self.dof + 1])
        self.kappa = np.concatenate([self.kappa[:1], self.kappa + 1])




class StudentT(Distribution):
    """ Generalized Student t distribution
    https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution
    This setting corresponds to select
      1: Gaussian distribution as a likelihood
      2: normal-Gamma distribution as a prior for Gaussian
    """

    def __init__(self, mu=0, kappa=1, alpha=1, beta=1):
        self.mu0 = np.array([mu])
        self.kappa0 = np.array([kappa])
        self.alpha0 = np.array([alpha])
        self.beta0 = np.array([beta])
        # We need the following lines to prevent "outside defined warning"
        self.muT = self.mu0.copy()
        self.kappaT = self.kappa0.copy()
        self.alphaT = self.alpha0.copy()
        self.betaT = self.beta0.copy()

    def reset_params(self):
        self.muT = self.mu0.copy()
        self.kappaT = self.kappa0.copy()
        self.alphaT = self.alpha0.copy()
        self.betaT = self.beta0.copy()

    def pdf(self, x):
        """ Probability Density Function
        """
        return stats.t.pdf(
            x,
            loc=self.muT,
            df=2 * self.alphaT,
            scale=np.sqrt(self.betaT * (self.kappaT + 1) / (self.alphaT * self.kappaT)),
        )

    def update_params(self, x):
        """Update Sufficient Statistcs (Parameters)
        To understand why we use this, see e.g.
        Conjugate Bayesian analysis of the Gaussian distribution, Kevin P. Murphy∗
        https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        3.5 Posterior predictive
        """
        self.betaT = np.concatenate(
            [
                self.beta0,
                (self.kappaT + (self.kappaT * (x - self.muT) ** 2) / (2 * (self.kappaT + 1))),
            ]
        )
        self.muT = np.concatenate([self.mu0, (self.kappaT * self.muT + x) / (self.kappaT + 1)])
        self.kappaT = np.concatenate([self.kappa0, self.kappaT + 1])
        self.alphaT = np.concatenate([self.alpha0, self.alphaT + 0.5])