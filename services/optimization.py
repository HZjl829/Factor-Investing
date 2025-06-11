import cvxpy as cp
import numpy as np
from scipy.stats import chi2
from scipy.stats import norm

def MVO(mu, Q):
    """
    #---------------------------------------------------------------------- 
    # Use this function to construct an example of a MVO portfolio.
    #
    # An example of an MVO implementation is given below. 

    #----------------------------------------------------------------------
    """

    # Find the total number of assets
    n = len(mu)

    # Set the target as the average expected return of all assets
    targetRet = np.mean(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using CVXPY
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb])
    prob.solve(verbose=False)
    return x.value


def risk_parity(Q, c=1.0, solver=cp.SCS):
    """
    Solve min_y ½ yᵀQy - c ∑ᵢ ln(yᵢ)  s.t. y > 0,
    then return w = y / sum(y).
    """
    n = Q.shape[0]
    y = cp.Variable(n, pos=True)
    obj = 0.5*cp.quad_form(y, Q) - c*cp.sum(cp.log(y))
    prob = cp.Problem(cp.Minimize(obj),
                      [y>= 0])
    prob.solve(solver=solver, verbose=False)
    y_opt = y.value
    return y_opt / np.sum(y_opt)



def robust_MVO_box(mu, Q, T,
                   alpha=0.95,
                   targetRet=None,
                   solver=cp.SCS):
    """
    Robust MVO under *box* uncertainty in mu:
        min_x  x' Q x
        s.t.   mu' x  -  delta' |x|  >= targetRet
               sum(x)==1, x>=0

    where delta_i = z_{alpha} * sqrt(Q_{ii}/T)

    Args:
      mu        : (n,) or (n,1) array of expected returns
      Q         : (n,n) covariance matrix
      T         : # observations used to estimate standard errors
      alpha     : confidence level (so z_alpha = norm.ppf(alpha))
      targetRet : required return (defaults to mean(mu))
      solver    : CVXPY solver to use

    Returns:
      x.value   : (n,) robust portfolio weights
    """
    mu = mu.flatten()
    n  = len(mu)

    # 1) default target = average expected return
    if targetRet is None:
        targetRet = float(mu.mean())

    # 2) compute per‐asset standard errors sqrt(Q_ii / T)
    theta_half = np.sqrt(np.diag(Q) / float(T))

    # 3) box‐uncertainty radius: z‐score for N(0,1)
    z_alpha = norm.ppf(alpha)

    # 4) build delta vector
    delta = z_alpha * theta_half   # shape (n,)

    # 5) decision variable
    x = cp.Variable(n)

    # 6) constraints
    constraints = [
        mu @ x
          - delta @ cp.abs(x)    # δᵀ|x|
          >= targetRet,
        cp.sum(x) == 1,
        x >= 0
    ]

    # 7) objective + solve
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, Q)), constraints)
    prob.solve(solver=solver, verbose=False)

    print("Status:", prob.status)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise ValueError(f"Robust box‐MVO infeasible (α={alpha:.2f}, R={targetRet:.6g})")

    return x.value





