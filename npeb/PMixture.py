import numpy as np
import pandas as pd
import itertools
from scipy.stats import poisson

try:
    from mosek.fusion import *
except:
    print("Warning: Could not load module named mosek.fusion")

def pois_pmf(X, atoms):
    if len(X.shape) > 1:
        X = np.concatenate(X)
    if len(atoms.shape) > 1:
        atoms = np.concatenate(atoms)
    X = X[:, np.newaxis]
    atoms = atoms[np.newaxis, :]
    return(poisson.pmf(X, atoms))

# t <= log(x), x>=0
def log(M, t, x):
    M.constraint(Expr.hstack(x, 1, t), Domain.inPExpCone())

def solve_weights_mosek(A, nks, **solver_params):
    """
    given:
       an n x m kernel A of probabilities
       a length n vector nks of counts
    return:
       the weights w_1,...,w_m maximizing 
       sum_{i=1}^n nks_i log(Aw)_i
       via an exponential cone program
    """
    n,m = A.shape
    M = Model()

    t = M.variable(n)
    u = M.variable(n, Domain.greaterThan(0.0))
    w = M.variable(m, Domain.inRange(0.0, 1.0))

    # exponential cone constraints
    for i in range(n):
        log(M, t.index(i), u.index(i))
    M.constraint(Expr.sum(w), Domain.equalsTo(1.0))
    for i in range(n):
        M.constraint(Expr.sub(Expr.dot(A[i], w), u.index(i)), 
                     Domain.equalsTo(0.0))

    # Set the objective function to sum_i t_i * nks_i
    M.objective("obj", ObjectiveSense.Maximize, Expr.dot(t, nks))

    # Solve
    M.solve()
    return w.level()

def solve_weights_mosek_heteroskedastic(B, **solver_params):
    """
    given:
       an n x m kernel A of probabilities
       a length n vector nks of counts
    return:
       the weights w_1,...,w_m maximizing 
       sum_{i=1}^n nks_i log(Aw)_i
       via an exponential cone program
    """
    n,m = B.shape
    M = Model()

    t = M.variable(n)
    u = M.variable(n, Domain.greaterThan(0.0))
    w = M.variable(m, Domain.inRange(0.0, 1.0))

    # exponential cone constraints
    for i in range(n):
        log(M, t.index(i), u.index(i))
    M.constraint(Expr.sum(w), Domain.equalsTo(1.0))
    for i in range(n):
        M.constraint(Expr.sub(Expr.dot(B[i,:], w), u.index(i)), 
                     Domain.equalsTo(0.0))

    M.objective("obj", ObjectiveSense.Maximize, Expr.sum(t))

    # Solve
    M.solve()
    return w.level()

class PMixture:
    """
    Kiefer-Wolfowitz nonparametric maximum likelihood estimation
    (NPMLE) for Poisson mixtures.

    ----------------------------OPTIONS-----------------------------

    atoms_init : array, default None
        Specify a set of atoms to use in the discretization
    
    --------------------------ATTRIBUTES----------------------------

    m : int (number of atoms)

    n : int (number of training obs)

    weights : ndarray of shape (m,)

    atoms : ndarray of shape (m,)

    XTrain : ndarray of shape (n,)
    """

    def __init__(self, atoms_init=None):
        self.atoms_init = atoms_init
        
    def get_params(self):
        return(self.atoms, self.weights)

    def set_params(self, atoms, weights):
        self.atoms, self.weights = atoms, weights

    def initialize_atoms_subsample(self, X, n_atoms):
        self.atoms_init = X[np.random.choice(X.shape[0], 
                                             size=n_atoms, 
                                             replace=False)]

    def initialize_atoms_grid(self, X, n_atoms):
        self.atoms_init = np.linspace(0, X.max()+1, n_atoms)

    def fit(self, X, **solver_params):
        self.n = len(X)
        self.XTrain = X

        if self.atoms_init is None:
            self.initialize_atoms_grid(X, max(50, int(self.n**.5)))
        atoms = self.atoms_init
        
        value_cts = pd.value_counts(X).sort_index()
        vals, cts = np.array(value_cts.keys()), list(value_cts.values)
            
        A = pois_pmf(vals, atoms)
        weights = solve_weights_mosek(A, cts, **solver_params)
        self.set_params(atoms, np.maximum(weights, 0))
    
    def fit_heteroskedastic(self, X, lams, **solver_params):
        self.n = len(X)
        self.XTrain = X

        if self.atoms_init is None:
            self.initialize_atoms_grid(X, max(50, int(self.n**.5)))
        atoms = self.atoms_init
        
        A = pois_pmf(X, atoms)
        A = np.divide(A, np.exp(np.outer(lams-1,atoms)))
        A = np.multiply(A, np.power(lams,X).reshape(-1,1))
        
        weights = solve_weights_mosek_heteroskedastic(A, **solver_params)
        self.set_params(atoms, np.maximum(weights, 0))
    
    def fit_bivariate_heteroskedastic(self, X, lams, **solver_params):
        self.n, self.d = X.shape
        self.XTrain = X
        
        # TODO: need bivariate atoms
        if self.atoms_init is None:
            self.initialize_atoms_grid(X, max(50, int(self.n**.5)))
            
        atoms = np.array(list(itertools.product(self.atoms_init,self.atoms_init)))
        
        A0 = pois_pmf(X[:,0], atoms[:,0])
        A0 = np.divide(A0, np.exp(np.outer(lams[:,0]-1,atoms[:,0])))
        A0 = np.multiply(A0, np.power(lams[:,0],X[:,0]).reshape(-1,1))
        
        A1 = pois_pmf(X[:,1], atoms[:,1])
        A1 = np.divide(A1, np.exp(np.outer(lams[:,1]-1,atoms[:,1])))
        A1 = np.multiply(A1, np.power(lams[:,1],X[:,1]).reshape(-1,1))
        
        A = np.multiply(A0,A1)
        
        weights = solve_weights_mosek_heteroskedastic(A, **solver_params)
        self.set_params(atoms, np.maximum(weights, 0))

    def posterior_mean(self, X):
        return self.posterior_mean_heteroskedastic(X, np.ones(len(X)))
    
    def posterior_mean_heteroskedastic(self, X, lams):
        a, w = self.get_params()
        K = pois_pmf(X, a)
        K = np.divide(K, np.exp(np.outer(lams-1,a)))
        K = np.multiply(K, np.power(lams,X).reshape(-1,1))
        return((K@(a*w)) / (K@w))
    
    def posterior_mean_bivariate_heteroskedastic(self, X, lams):
        a, w = self.get_params()
        A0 = pois_pmf(X[:,0], a[:,0])
        A0 = np.divide(A0, np.exp(np.outer(lams[:,0]-1,a[:,0])))
        A0 = np.multiply(A0, np.power(lams[:,0],X[:,0]).reshape(-1,1))
        
        A1 = pois_pmf(X[:,1], a[:,1])
        A1 = np.divide(A1, np.exp(np.outer(lams[:,1]-1,a[:,1])))
        A1 = np.multiply(A1, np.power(lams[:,1],X[:,1]).reshape(-1,1))
        
        K = np.multiply(A0,A1)
        aw = np.multiply(a,w.reshape(-1,1))
        return np.divide(K@aw,(K@w).reshape(-1,1))
