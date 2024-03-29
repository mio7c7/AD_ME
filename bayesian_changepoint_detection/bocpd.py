import abc
from river import base
import numpy as np
import scipy.stats as ss
from itertools import islice
from numpy.linalg import inv
from functools import partial

class ChangePointDetector(base.Base):
    """
    An abstract class for change point detection methods. Use this class to be able to run test using the tcpd benchmark.
    """

    def __init__(self, **kwargs):
        self._change_point_detected = False
        self._change_point_score = 0.0

    def _reset(self):
        """
        Reset the change detector.
        """
        self._change_point_detected = False
        self._change_point_score = 0.0

    @property
    def change_point_detected(self) -> bool:
        """
        Returns True if a change point was detected.
        """
        return self._change_point_detected

    @property
    def change_point_score(self) -> float:
        """
        Returns the change point score.
        """
        return self._change_point_score

    @abc.abstractmethod
    def update(self, x, t) -> "ChangePointDetector":
        """
        Update the change point detector with a single data point.
        :param x: Input data point.
        :param t: Time step. STARTING FROM 1.
        :return: self
        """

    @abc.abstractmethod
    def is_multivariate(self):
        """
        Returns True if the change point detector can handle multivariate input sequences.
        """

class MultivariateT():
    def __init__(
        self,
        dims: int = 1,
        dof: int = 0,
        kappa: int = 1,
        mu: float = -1,
        scale: float = -1,
    ):
        """
        Generate a new predictor using the multivariate student T distribution as the posterior predictive.
        This implies a multivariate Gaussian distribution on the data, a Wishart prior on the precision,
        and a Gaussian prior on the mean.
        Implementation based on Haines, T.S., Gaussian Conjugate Prior Cheat Sheet.
        Parameters:
        dof - The degrees of freedom on the prior distribution of the precision (inverse covariance)
        kappa - The number of observations we've already seen
        mu - The mean of the prior distribution on the mean
        scale - The mean of the prior distribution on the precision
        dims - The number of variables
        """
        # We default to the minimum possible degrees of freedom, which is 1 greater than the dimensionality
        if dof == 0:
            dof = dims + 1
        if mu == -1:
            mu = [0] * dims # default mean
        else:
            mu = [mu] * dims

        # The default covariance is the identity matrix, so the scale is also the inverse of the identity.
        if scale == -1:
            scale = np.identity(dims)
        else:
            scale = np.identity(scale)

        # Track time
        self.t = 0

        # number of variables
        self.dims = dims

        # Each parameter is a vector of size 1 x t, where t is time. Therefore each vector grows with each update.
        self.dof = np.array([dof])
        self.kappa = np.array([kappa])
        self.mu = np.array([mu])
        self.scale = np.array([scale])

    def pdf(self, data: np.array):
        """
        Returns the probability of the observed data under the current and historical parameters
        Parmeters:
            data - the datapoints to be evaualted (shape: 1 x D vector)
        """
        self.t += 1
        t_dof = self.dof - self.dims + 1
        expanded = np.expand_dims(
            (self.kappa * t_dof) / (self.kappa + 1), (1, 2))
        ret = np.empty(self.t)
        try:
            # This can't be vectorised due to https://github.com/scipy/scipy/issues/13450
            for i, (df, loc, shape) in islice(
                enumerate(zip(t_dof, self.mu, inv(
                    expanded * self.scale))), self.t
            ):
                ret[i] = ss.multivariate_t.pdf(
                    x=data, df=df, loc=loc, shape=shape)
        except AttributeError:
            raise Exception(
                "You need scipy 1.6.0 or greater to use the multivariate t distribution"
            )
        return ret

    def update_theta(self, data: np.array, **kwargs):
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


class StudentT():
    def __init__(
        self, alpha: float = 0.1, beta: float = 0.1, kappa: float = 1, mu: float = 0
    ):
        """
        StudentT distribution except normal distribution is replaced with the student T distribution
        https://en.wikipedia.org/wiki/Normal-gamma_distribution
        Parameters:
            alpha - alpha in gamma distribution prior
            beta - beta inn gamma distribution prior
            mu - mean from normal distribution
            kappa - variance from normal distribution
        """

        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data: np.array):
        """
        Return the pdf function of the t distribution
        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        return ss.t.pdf(
            x=data,
            df=2 * self.alpha,
            loc=self.mu,
            scale=np.sqrt(self.beta * (self.kappa + 1) /
                          (self.alpha * self.kappa)),
        )

    def update_theta(self, data: np.array, **kwargs):
        """
        Performs a bayesian update on the prior parameters, given data
        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        muT0 = np.concatenate(
            (self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1))
        )
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.0))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate(
            (
                self.beta0,
                self.beta
                + (self.kappa * (data - self.mu) ** 2) /
                (2.0 * (self.kappa + 1.0)),
            )
        )

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0


class BOCPD(ChangePointDetector):

    def __init__(self, threshold, delay, lmt, **kwargs):
        super().__init__(**kwargs)
        self.hazard_function = partial(self.constant_hazard, 5760)
        self.log_likelihood_class = MultivariateT(dims=2, dof=2, kappa=10)

        self.len_data_estimate = lmt+1
        self.maxes = np.zeros(self.len_data_estimate)#
        self.R = np.zeros((self.len_data_estimate, self.len_data_estimate))
        self.R[0, 0] = 1
        self.threshold = threshold
        self.delay = delay

    def update(self, x, t, reseted, NW) -> "ChangePointDetector":
        self._change_point_detected = False

        # Compute the predictive probabilities of the data x
        predprobs = self.log_likelihood_class.pdf(x)

        if reseted:
            t = t + NW
        # Evaluate the hazard function for this interval
        H = self.hazard_function(np.array(range(t + 1)))
        m = self.R[0: t + 1, t]
        k = self.R[0: t + 1, t] * predprobs * (1 - H)
        # Evaluate the growth probabilities
        # Shift the probabilities down and to the right, scaled by the hazard function and the predictive probabilities.
        self.R[1: t + 2, t + 1] = self.R[0: t + 1, t] * predprobs * (1 - H)

        # Evaluate the probability that there *was* a changepoint and we're accumulating the mass back down at r = 0.
        self.R[0, t + 1] = np.sum(self.R[0: t + 1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical stability.
        self.R[:, t + 1] = self.R[:, t + 1] / np.sum(self.R[:, t + 1])

        # Update the parameter sets for each possible run length.
        self.log_likelihood_class.update_theta(x, t=t)

        # Store the index with the maximum probability
        self.maxes[t] = self.R[:, t].argmax()

        # Check if a change point has been detected
        # if self.maxes[t-1] - self.maxes[t] > self.threshold:
        #     self._change_point_detected = True

        if t > self.delay and self.R[self.delay, t] > self.threshold:
            self._change_point_detected = True

        # Return the updated ChangePointDetector object
        return self

    def constant_hazard(self, lam, r):
        """
        Hazard function for bayesian online learning
        Arguments:
            lam - inital prob
            r - R matrix
        """
        return 1 / lam * np.ones(r.shape)

    def _reset(self):
        super()._reset()
        self.maxes = np.zeros(self.len_data_estimate)  #
        self.R = np.zeros((self.len_data_estimate, self.len_data_estimate))
        self.R[0, 0] = 1
        self.log_likelihood_class = MultivariateT(dims=2, dof=2, kappa=10)

    def prune(self, NW):
        old = self.log_likelihood_class
        orgR = self.R
        self.log_likelihood_class = MultivariateT(dims=2, dof=2, kappa=10)
        # keep the NW samples value from the last iteration
        self.log_likelihood_class.mu = old.mu[-NW-1:,:]
        self.log_likelihood_class.scale = old.scale[-NW-1:,:,:]
        self.log_likelihood_class.dof = old.dof[:NW+1]
        self.log_likelihood_class.kappa = old.kappa[:NW+1]
        self.log_likelihood_class.t = NW
        self.maxes = np.zeros(self.len_data_estimate+NW)  #
        self.R = np.zeros((self.len_data_estimate+NW, self.len_data_estimate+NW))
        l = orgR[0: NW+1,-1]
        sum = np.sum(l)
        l = l / sum
        self.R[0: NW+1, NW] = l

    def is_multivariate(self):
        return True


class BOCPD_UNI(ChangePointDetector):

    def __init__(self, threshold, delay, lmt, **kwargs):
        super().__init__(**kwargs)
        self.hazard_function = partial(self.constant_hazard, 5760)
        self.log_likelihood_class = StudentT(kappa=10)

        self.len_data_estimate = lmt+1
        self.maxes = np.zeros(self.len_data_estimate)#
        self.R = np.zeros((self.len_data_estimate, self.len_data_estimate))
        self.R[0, 0] = 1
        self.threshold = threshold
        self.delay = delay

    def update(self, x, t, reseted, NW) -> "ChangePointDetector":
        self._change_point_detected = False

        # Compute the predictive probabilities of the data x
        predprobs = self.log_likelihood_class.pdf(x)

        if reseted:
            t = t + NW
        # Evaluate the hazard function for this interval
        H = self.hazard_function(np.array(range(t + 1)))
        m = self.R[0: t + 1, t]
        k = self.R[0: t + 1, t] * predprobs * (1 - H)
        # Evaluate the growth probabilities
        # Shift the probabilities down and to the right, scaled by the hazard function and the predictive probabilities.
        self.R[1: t + 2, t + 1] = self.R[0: t + 1, t] * predprobs * (1 - H)

        # Evaluate the probability that there *was* a changepoint and we're accumulating the mass back down at r = 0.
        self.R[0, t + 1] = np.sum(self.R[0: t + 1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical stability.
        self.R[:, t + 1] = self.R[:, t + 1] / np.sum(self.R[:, t + 1])

        # Update the parameter sets for each possible run length.
        self.log_likelihood_class.update_theta(x, t=t)

        # Store the index with the maximum probability
        self.maxes[t] = self.R[:, t].argmax()

        # Check if a change point has been detected
        # if self.maxes[t-1] - self.maxes[t] > self.threshold:
        #     self._change_point_detected = True

        if t > self.delay and self.R[self.delay, t] > self.threshold:
            self._change_point_detected = True

        # Return the updated ChangePointDetector object
        return self

    def constant_hazard(self, lam, r):
        """
        Hazard function for bayesian online learning
        Arguments:
            lam - inital prob
            r - R matrix
        """
        return 1 / lam * np.ones(r.shape)

    def _reset(self):
        super()._reset()
        self.maxes = np.zeros(self.len_data_estimate)  #
        self.R = np.zeros((self.len_data_estimate, self.len_data_estimate))
        self.R[0, 0] = 1
        self.log_likelihood_class = StudentT(kappa=10)

    def prune(self, NW):
        old = self.log_likelihood_class
        orgR = self.R
        self.log_likelihood_class = StudentT(kappa=10)
        # keep the NW samples value from the last iteration
        self.log_likelihood_class.mu = old.mu[-NW-1:]
        self.log_likelihood_class.alpha = old.alpha[-NW-1:]
        self.log_likelihood_class.beta = old.beta[:NW+1]
        self.log_likelihood_class.kappa = old.kappa[:NW+1]
        self.log_likelihood_class.t = NW
        self.maxes = np.zeros(self.len_data_estimate+NW)  #
        self.R = np.zeros((self.len_data_estimate+NW, self.len_data_estimate+NW))
        l = orgR[0: NW+1,-1]
        sum = np.sum(l)
        l = l / sum
        self.R[0: NW+1, NW] = l

    def is_multivariate(self):
        return False