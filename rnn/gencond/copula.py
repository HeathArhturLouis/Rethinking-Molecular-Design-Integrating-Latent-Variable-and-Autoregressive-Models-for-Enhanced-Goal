import numpy as np

from abc import ABC, abstractmethod
from scipy.stats import norm, multivariate_normal

from sklearn.mixture import BayesianGaussianMixture

normal_cdf = norm.cdf
normal_icdf = norm.ppf


class Marginal(ABC):
    """ Marginal distributions for multivariate copula """
    
    @abstractmethod
    def fit(self, data):
        ''' Fit marginal to data '''
        pass
    
    @abstractmethod
    def transform(self, data):
        ''' Transform from sample space to z-space '''
        pass
    
    @abstractmethod
    def inverse_transform(self, z):
        ''' Transform back from z-space to sample space '''
        pass
    
    def sample(self, shape=(1,)):
        ''' Draw samples from marginal '''
        return self.inverse_transform(np.random.randn(*shape))


class DiscreteMarginal(Marginal):
    """ Discrete marginal distribution """
    
    def __init__(self, smoothing=1.0, pad_ratio=2.0):
        self.smoothing = smoothing
        self.pad_ratio = pad_ratio
    
    def fit(self, data):
        max_val = np.max(data)
        padded_max = int(max_val * self.pad_ratio)
        values = np.bincount(np.concatenate((np.array(data, dtype=int), np.arange(padded_max)))) + (self.smoothing - 1)
        total = values.sum()
        rhs = values.cumsum() / total
        self.lhs = np.concatenate(([0], rhs[:-1]))
        self.rhs = rhs
        self.binwidth = self.rhs - self.lhs
        return self
        
    def transform(self, data):
        data = np.array(data, dtype=int, copy=False)
        unit = self.lhs[data] + self.binwidth[data]*np.random.rand(*data.shape)
        return normal_icdf(unit)

    def inverse_transform(self, z):
        value = normal_cdf(z)
        which = (value[..., None] >= self.rhs).sum(-1)
        return which


class ContinuousMarginal(Marginal):
    """ Continuous marginal distribution; implemented as mixture of Gaussians """
    
    def __init__(self, K=8, subsample=100000, steps=2000):
        self.K = K
        self.subsample = subsample
        self.steps = steps
        
    def fit(self, data):
        assert data.ndim == 1 # only run on univariate marginals
        domain = data.min(), data.max()
        if self.subsample is not None and self.subsample < len(data):
            data = data[np.random.randint(0, len(data), (self.subsample, ))]
        model = BayesianGaussianMixture(n_components=self.K, tol=0.01, max_iter=250)
        model.fit(data.reshape(-1, 1))
        self.weights = model.weights_
        self.means = model.means_[:,0]
        self.scales = np.sqrt(model.covariances_[:,0,0])
        self.minval = (self.means - 4.5*self.scales).min()
        self.maxval = (self.means + 4.5*self.scales).max()
        self._loc = np.linspace(self.minval, self.maxval, self.steps)
        self._val = self.cdf(self._loc)
        return self
    
    def cdf(self, data):
        return np.dot((normal_cdf(((data[..., None] - self.means)/self.scales))), self.weights)

    def transform(self, data):
        unit = self.cdf(np.clip(data, self.minval, self.maxval))
        assert np.isfinite(unit).all()
        return normal_icdf(unit)

    def inverse_transform(self, z):
        # TODO: this is approximate, and based off a piecewise linear approximation.
        #       Consider replacing with a local optimization routine if necessary
        unit = normal_cdf(z)
        compare = 1.0*(self._val < unit[..., None])
        # TODO: this will fail if values of z fall outside (minval, maxval)
        idx = compare.argmin(-1)-1
        w = (unit - self._val[idx]) / (self._val[idx+1] - self._val[idx])
        assert (self._val[idx] <= unit).all()
        assert (self._val[idx+1] >= unit).all()
        est = self._loc[idx] + w * (self._loc[idx+1] - self._loc[idx])
        return est


class GaussianCopula(object):
    """ Class for defining a joint distribution, given marginals. 
        This is basically a Gaussian copula model. """
    
    def __init__(self, marginals, names=None):
        """ Takes a list of marginal distributions """
        self.marginals = marginals
        if names is None:
            self.names = np.arange(len(marginals))
        else:
            self.names = names
        
    def _fit_marginals(self, data, verbose=False):
        assert data.ndim == 2
        assert data.shape[1] == len(self.marginals)
        for i in range(data.shape[1]):
            if verbose:
                print("Fitting %s '%s', (%d of %d)" % \
                      (self.marginals[i].__class__.__name__, self.names[i], i+1, data.shape[1]))
            self.marginals[i].fit(data[:,i])
            
    def fit(self, data, fit_marginals=True, verbose=False):
        assert data.ndim == 2
        assert data.shape[1] == len(self.marginals)
        if fit_marginals:
            self._fit_marginals(data, verbose)
        if verbose:
            print("Fitting correlation matrix ...")
        transformed_data = self.transform(data)
        self.cov = np.corrcoef(transformed_data.T)
        self._dist = multivariate_normal(mean=np.zeros(data.shape[1]), cov=self.cov)

    def transform(self, data):
        return np.stack([self.marginals[i].transform(data[:,i]) for i in range(data.shape[1])]).T

    def inverse_transform(self, z):
        return np.stack([self.marginals[i].inverse_transform(z[:,i]) for i in range(z.shape[1])]).T

    def sample(self, shape=(1,)):
        return self._dist.rvs(shape)
    
    def log_prob(self, z):
        return self._dist.logpdf(z)
        
    # TODO this will need additional methods later for doing conditional generation
