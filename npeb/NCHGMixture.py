import numpy as np
from scipy.special import gammaln, logsumexp

try:
    from mosek.fusion import *
except:
    print("Warning: Could not load module named mosek.fusion")

try:
    import cvxpy as cvx
except:
    print("Warning: Could not load module named cvxpy")

def log_nchg_pmf(table, theta):
    """
    given:
       an n x 4 integer ndarray `table` whose rows are
           (Z1, X1, Z2, X2) = (treatment visits, treatment non-visits,
                               control visits, control non-visits),
       a length-G ndarray `theta` of log-odds-ratio values
    return:
       an n x G ndarray A of log-probabilities, where
       A[i, g] = log p(Z1_i | theta[g]) and p(. | theta) is Fisher's
       noncentral hypergeometric pmf for the 2x2 table conditional on
       its margins, i.e. treatment total n1 = Z1 + X1, control total
       n2 = Z2 + X2, total successes s = Z1 + Z2, and odds ratio e^theta:

         p(z | theta) = C(n1, z) C(n2, s - z) e^{theta z} / P0(theta),

       supported on max(0, s - n2) <= z <= min(n1, s).
    """
    Z1, X1, Z2, X2 = (table[:, 0], table[:, 1], table[:, 2], table[:, 3])
    n1, n2, s = Z1 + X1, Z2 + X2, Z1 + Z2
    n, G = table.shape[0], len(theta)
    A = np.empty((n, G))
    for i in range(n):
        lo, hi = max(0, s[i] - n2[i]), min(n1[i], s[i])
        zs = np.arange(lo, hi + 1)
        ## log binomial coefficients log C(n1, z) + log C(n2, s - z)
        lb = (gammaln(n1[i] + 1) - gammaln(zs + 1) - gammaln(n1[i] - zs + 1)
              + gammaln(n2[i] + 1) - gammaln(s[i] - zs + 1)
              - gammaln(n2[i] - s[i] + zs + 1))
        ## unnormalized log-pmf over the support at every theta, then
        ## subtract the log-partition log P0(theta) columnwise
        M = lb[:, np.newaxis] + np.outer(zs, theta)
        k = int(np.searchsorted(zs, Z1[i]))
        A[i] = M[k] - logsumexp(M, axis=0)
    return(A)

def normal_pdf(x, loc, scale):
    return(np.exp(-0.5 * ((x - loc) / scale)**2) / (scale * np.sqrt(2 * np.pi)))

def solve_weights_cvx(A, **solver_params):
    """
    given:
       an n x m kernel A of marginal likelihoods, scaled by row
    return:
       the weights w_1,...,w_m maximizing
        sum_{i=1}^n log(Aw)_i
       using cvxpy
    """
    n, m = A.shape
    w = cvx.Variable(m)
    constraints = [w >= 0, cvx.sum(w) == 1]
    obj = cvx.Maximize(cvx.sum(cvx.log(A @ w)))
    prob = cvx.Problem(obj, constraints)
    prob.solve(**solver_params)
    return(w.value)

def solve_weights_mosek(A, **solver_params):
    """
    given:
       an n x m kernel A of marginal likelihoods, scaled by row
    return:
       the weights w_1,...,w_m maximizing
        sum_{i=1}^n log(Aw)_i
       via an exponential cone program in mosek
    """
    n, m = A.shape
    M = Model()

    t = M.variable(n)
    u = M.variable(n, Domain.greaterThan(0.0))
    w = M.variable(m, Domain.inRange(0.0, 1.0))

    ## exponential cone constraints: t[i] <= log(u[i]), u[i] >= 0
    for i in range(n):
        M.constraint(Expr.hstack(u.index(i), 1, t.index(i)),
                     Domain.inPExpCone())
    M.constraint(Expr.sum(w), Domain.equalsTo(1.0))
    for i in range(n):
        M.constraint(Expr.sub(Expr.dot(A[i], w), u.index(i)),
                     Domain.equalsTo(0.0))

    M.objective("obj", ObjectiveSense.Maximize, Expr.sum(t))

    for k, v in solver_params.items():
        M.setSolverParam(k, v)
    M.solve()
    return w.level()

class NCHGMixture:
    """
    Kiefer-Wolfowitz nonparametric maximum likelihood estimation
    (NPMLE) for noncentral hypergeometric mixtures, over a sieve of
    Gaussian components Normal(mu, component_std) on the log-odds-ratio.

    Each observation is a 2x2 table (treatment / control by
    success / failure); conditional on its margins, the treatment
    success count follows Fisher's noncentral hypergeometric
    distribution whose log-odds-ratio theta ~ G. Following Empirikos,
    G is estimated over the mixture class {sum_j w_j Normal(mu_j, s)}.

    ----------------------------OPTIONS-----------------------------

    component_std : float, default 0.1
        Standard deviation s of the Gaussian mixture components.

    atoms_init : array, default None
        Component means mu_j to use in the discretization.

    theta_step : float, default 0.005
        Spacing of the quadrature grid over theta.

    n_sd : float, default 8.0
        The quadrature grid spans n_sd component_std's beyond the atoms.

    --------------------------ATTRIBUTES----------------------------

    m : int (number of atoms)

    n : int (number of training obs)

    weights : ndarray of shape (m,)

    atoms : ndarray of shape (m,)  (Gaussian component means)

    ZTrain : ndarray of shape (n, 4)  (rows (Z1, X1, Z2, X2))
    """

    def __init__(self, component_std=0.1, atoms_init=None,
                 theta_step=0.005, n_sd=8.0):
        self.component_std = component_std
        self.atoms_init = atoms_init
        self.theta_step = theta_step
        self.n_sd = n_sd

    def get_params(self):
        return(self.atoms, self.weights)

    def set_params(self, atoms, weights):
        self.atoms, self.weights = atoms, weights

    def initialize_atoms_grid(self, lo=-3., hi=3., step=0.01):
        self.atoms_init = np.round(np.arange(lo, hi + step / 2, step), 6)

    def _theta_grid(self, atoms):
        pad = self.n_sd * self.component_std
        return(np.arange(atoms.min() - pad, atoms.max() + pad + self.theta_step / 2,
                         self.theta_step))

    def _kernel(self, table, atoms):
        """
        return the n x m marginal-likelihood kernel A with
        A[i, j] = int p(Z1_i | theta) Normal(theta | mu_j, s) dtheta,
        computed on a fine theta grid (tail mass ~ 1 for n_sd >= 8),
        together with the grid, its spacing, and the smoothing matrix.
        """
        theta = self._theta_grid(atoms)
        h = self.theta_step
        L = np.exp(log_nchg_pmf(table, theta))                 # n x G
        S = normal_pdf(theta[:, np.newaxis], atoms[np.newaxis, :],
                       self.component_std)                     # G x m
        A = h * (L @ S)
        return(A, theta, h, L)

    def fit(self, table, weight_thresh=0., row_condition=True,
            solver='mosek', **solver_params):
        """
        given:
           an n x 4 integer ndarray `table` of (Z1, X1, Z2, X2) rows
           weight_thresh : float, default 0
               threshold value for discretized NPMLE weights
           row_condition : bool, default True
               divide each row of the kernel by its max before solving
           solver : 'mosek' or 'cvx'

        Solve the discretized NPMLE over the fixed grid of Gaussian
        components (mixture weights only). Updates self.atoms and
        self.weights.
        """
        table = np.asarray(table)
        self.n = table.shape[0]
        self.ZTrain = table

        if self.atoms_init is None:
            self.initialize_atoms_grid()
        atoms = self.atoms_init

        print('Computing kernel matrix:', end=' ')
        A, self.theta, self.h, _ = self._kernel(table, atoms)
        if row_condition:
            A = A / A.max(1, keepdims=True)
        print('done.')

        print('Solving for discretized NPMLE:', end=' ')
        if solver == 'mosek':
            w = solve_weights_mosek(A, **solver_params)
        elif solver == 'cvx':
            w = solve_weights_cvx(A, **solver_params)
        print('done.')

        w = np.maximum(w, 0.)
        atoms = atoms[w > weight_thresh]
        weights = w[w > weight_thresh]
        weights /= np.sum(weights)
        self.set_params(atoms, weights)

    def score(self, table):
        """
        return the average marginal log-likelihood across the rows of
        `table` under the fitted prior.
        """
        A, _, _, _ = self._kernel(np.asarray(table), self.atoms)
        return(np.mean(np.log(A.dot(self.weights))))

    def prior_pdf(self, theta):
        """density of the fitted prior Ghat_n on a grid `theta`."""
        a, w = self.get_params()
        return(normal_pdf(np.atleast_1d(theta)[:, np.newaxis],
                          a[np.newaxis, :], self.component_std) @ w)

    def prior_mean(self):
        a, w = self.get_params()
        return(a @ w)

    def prior_std(self):
        a, w = self.get_params()
        return(np.sqrt(self.component_std**2 + (a**2 @ w) - (a @ w)**2))

    def prior_quantile(self, u, n_grid=20001):
        """quantile function of the fitted prior Ghat_n at levels `u`."""
        pad = self.n_sd * self.component_std
        a, _ = self.get_params()
        grid = np.linspace(a.min() - pad, a.max() + pad, n_grid)
        cdf = np.cumsum(self.prior_pdf(grid))
        cdf /= cdf[-1]
        return(np.interp(u, cdf, grid))

    def posterior_mean(self, table):
        """
        return the Bayes estimate (posterior mean of the log-odds-ratio
        theta) for each row of `table` under the fitted prior Ghat_n:
           E[theta | Z] = int theta p(Z | theta) dGhat_n(theta)
                        / int       p(Z | theta) dGhat_n(theta).
        """
        table = np.asarray(table)
        theta = self._theta_grid(self.atoms)
        L = np.exp(log_nchg_pmf(table, theta))                 # n x G
        rho = self.prior_pdf(theta)                            # G (Ghat_n density)
        num = (L * theta[np.newaxis, :]) @ rho
        den = L @ rho
        return(num / den)
