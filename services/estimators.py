import numpy as np
import gurobipy as gp
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
import pandas as pd


def mu_js_shrink(sample_mu, T, prior=None):
    """
    sample_mu : (n,1) array of sample means
    T         : # observations
    prior     : (n,1) vector to shrink toward (default = zeros)
    """
    n = len(sample_mu)
    if prior is None:
        prior = np.zeros_like(sample_mu)

    # magnitude of shrinkage
    # constant τ can be tuned or set to (n-2)/T
    tau = (n - 2) / float(T)
    w   = max(0, 1 - tau)   # shrink factor
    return w * sample_mu + (1 - w) * prior

def OLS(returns, factRet):
    """
    Basic OLS factor model.

    Returns:
      mu     : (n,1) expected returns
      Q      : (n,n) covariance matrix
      adj_R2 : (n,)  adjusted R² for each asset, 
               using p_eff = number of nonzero betas
    """
    # 1) dimensions
    T, p = factRet.shape
    assets = returns.columns
    n = len(assets)

    # 2) build design matrix + solve
    X = np.concatenate([np.ones((T,1)), factRet.values], axis=1)  # T×(p+1)
    B = np.linalg.solve(X.T @ X, X.T @ returns.values)            # (p+1)×n

    # 3) unpack intercept and betas
    a = B[0, :]          # (n,)
    V = B[1:, :]         # p×n

    # 4) residuals and idiosyncratic variances
    ep = returns.values - X @ B                                    # T×n
    sigma_ep = np.sum(ep**2, axis=0) / float(T - p - 1)             # (n,)
    D = np.diag(sigma_ep)                                           # n×n

    # 5) factor moments
    f_bar = factRet.mean(axis=0).values.reshape(p,1)                # p×1
    F     = factRet.cov().values                                    # p×p

    # 6) asset μ and Q
    mu = a.reshape(n,1) + V.T @ f_bar                               # n×1
    Q  = V.T @ F @ V + D                                            # n×n
    Q  = (Q + Q.T) / 2                                              # symmetrize

    # 7) compute adjusted R²
    #   SST and SSR per asset
    y_bar = returns.values.mean(axis=0)                             # (n,)
    SST   = np.sum((returns.values - y_bar)**2, axis=0)             # (n,)
    SSR   = np.sum(ep**2, axis=0)                                   # (n,)
    R2    = 1 - SSR / SST                                           # (n,)

    #   effective number of betas (count nonzero)
    p_eff = np.count_nonzero(V, axis=0)                             # (n,)

    #   adjusted R²
    adj_R2 = 1 - (1 - R2) * (T - 1) / (T - p_eff - 1)                # (n,)

    # flatten mu
    mu = mu.flatten()                                               # (n,)

    return mu, Q, adj_R2


def LASSO(returns, factRet):
    """
    Use this function for the LASSO model. 

    We find the optimal lambda using cross-validation if lambdas is None.
    Otherwise, we use the provided lambda.
    Returns:
      mu      : n-vector of expected returns
      Q       : n×n asset covariance matrix
      adj_R2  : n-vector of adjusted R² for each asset
    """
    # 1) align & clean
    # strip space in column names
    returns.columns = [col.strip() for col in returns.columns]
    factRet.columns = [col.strip() for col in factRet.columns]
    # print(factRet.columns)
    data = returns.join(factRet, how='inner').dropna()
    factor_cols = ['Mkt_RF','SMB','HML','RMW','CMA','Mom','ST_Rev','LT_Rev']
    F = data[factor_cols].values
    T, p = F.shape
    f_mean  = F.mean(axis=0)
    Sigma_f = np.cov(F, rowvar=False, ddof=1)

    assets = returns.columns
    N = len(assets)
    
    lambda_ = np.logspace(-5, -1, 50) 
    
   

    # if doing CV, set up a time-series splitter
    tscv = TimeSeriesSplit(n_splits=3) 

    # 3) storage
    alpha   = np.zeros(N)
    B       = np.zeros((N, p))
    eps_var = np.zeros(N)
    adj_R2  = np.zeros(N)

    # 4) fit each asset
    for i, asset in enumerate(assets):
        y = data[asset].values

        
        # pick lamdba via CV
        model = LassoCV(alphas=lambda_,
                        cv=tscv,
                        fit_intercept=True,
                        max_iter=10000)

        model.fit(F, y)
        

        # store the CV‐chosen coefficients
        coefs = model.coef_.copy()
        intercept = model.intercept_

        # count how many non-zeros
        p_eff = np.count_nonzero(coefs)
        if p_eff < 3:
            # force at least 3 non-zeros by trying smaller penalties
            for lam in sorted(lambda_):   # smallest diag first → least shrinkage
                tmp = Lasso(alpha=lam,
                            fit_intercept=True,
                            max_iter=10000).fit(F, y)
                if np.count_nonzero(tmp.coef_) >= 3:
                    coefs = tmp.coef_
                    intercept = tmp.intercept_
                    break
            # at this point `coefs` has ≥3 non-zeros (or is your best fallback)

        # now assign back
        alpha[i] = intercept
        B[i, :]  = coefs
        
        # residual stats
        resid      = y - model.predict(F)
        eps_var[i] = np.var(resid, ddof=1)

        # compute adjusted R2
        SSR = np.sum(resid**2)
        SST = np.sum((y - y.mean())**2)
        R2  = 1 - SSR/SST
        p_eff      = np.count_nonzero(model.coef_)
        # print("p_eff", p_eff)
        adj_R2[i]  = 1 - (1 - R2) * (T - 1) / (T - p_eff - 1)

    # 5) build mu & Q
    mu = alpha + B.dot(f_mean)                 # (n,)
    Q  = B.dot(Sigma_f).dot(B.T) + np.diag(eps_var)  # (n,n)


    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2
    return mu, Q, adj_R2

def BSS(returns, factRet, K = 5):
    """
    % Use this function for the BSS model. Note that you will not use
    % lambda in this model (lambda is for LASSO).
    %
    % You should use an optimizer to solve this problem. Be sure to comment
    % on your code to (briefly) explain your procedure.

    Inputs:
      - returns : pd.DataFrame (T × N) of each asset's excess monthly returns.
      - factRet : pd.DataFrame (T × 8) with factor columns
                  ['Mkt_RF','SMB','HML','RMW','CMA','Mom','ST_Rev','LT_Rev'].
      - lambda_  : not used here (for LASSO).
      - K        : int, maximum number of nonzero coefficients.
    Outputs:
      - mu : np.ndarray (N,) of expected returns from the factor model.
      - Q  : np.ndarray (N, N) covariance matrix from the factor model.
    """

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------
    # Align returns & factors by date, drop missing

    returns.columns = [col.strip() for col in returns.columns]
    factRet.columns = [col.strip() for col in factRet.columns]
    data = returns.join(factRet, how='inner').dropna()

    # Build regression matrix X = [1, factors]
    factor_cols = ['Mkt_RF','SMB','HML','RMW','CMA','Mom','ST_Rev','LT_Rev']
    F = data[factor_cols].values  # (T, 8)
    T, p = F.shape
    X = np.hstack([np.ones((T, 1)), F])  # (T, p+1)

    assets = returns.columns
    N = len(assets)

    # storage for outputs
    alpha = np.zeros(N)
    B     = np.zeros((N, p))
    eps_var = np.zeros(N)
    adj_R2 = np.zeros(N)

    # Solve a mixed-integer QP for each asset
    for i, asset in enumerate(assets):
        y = data[asset].values  # (T,)

        # Preliminary OLS to set a big-M bound
        ols_coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        M = np.max(np.abs(ols_coeffs)) * 2 + 1e-6

        # Build Gurobi model
        model = gp.Model()
        model.Params.OutputFlag = 0  # silent

        # continuous betas β₀...βₚ and binary selectors z₀...zₚ
        beta = model.addVars(p+1, lb=-M, ub=M, name="beta")
        z    = model.addVars(p+1, vtype=gp.GRB.BINARY, name="z")

        # Cardinality constraint: sum z_j ≤ K
        model.addConstr(gp.quicksum(z[j] for j in range(p+1)) <= K)

        # Big-M linking: β_j = 0 when z_j = 0
        for j in range(p+1):
            model.addConstr(beta[j] <=  M * z[j])
            model.addConstr(beta[j] >= -M * z[j])

        # 3e) Build quadratic objective: minimize ||y - Xβ||²
        Qmat = X.T @ X  # (p+1, p+1)
        cvec = -2 * (X.T @ y)  # (p+1,)
        obj = gp.QuadExpr()
        # β^T Q β term + linear term
        for j in range(p+1):
            obj.add(beta[j] * cvec[j])
            for k in range(p+1):
                obj.add(beta[j] * beta[k] * Qmat[j, k])

        model.setObjective(obj, gp.GRB.MINIMIZE)
        model.optimize()

        # 3f) Extract solution
        sol = np.array([beta[j].X for j in range(p+1)])
        alpha[i]   = sol[0]
        B[i, :]    = sol[1:]
        resid      = y - X.dot(sol)
        eps_var[i] = np.var(resid, ddof=1)

        # Compute R² and adjusted R²
        SSR = np.sum(resid**2)
        SST = np.sum((y - y.mean())**2)
        R2  = 1 - SSR/SST
        p_eff = np.count_nonzero(sol[1:])

        # Adjusted R² using the effective number of predictors
        adj_R2[i] = 1 - (1 - R2) * (T - 1) / (T - p_eff - 1)
        

    # Compute μ and Σ as in a factor model
    f_mean = F.mean(axis=0)                   # (8,)
    Sigma_f = np.cov(F, rowvar=False, ddof=1)  # (8,8)
    

    # print(B)
    mu = alpha + B.dot(f_mean)
    Q  = B.dot(Sigma_f).dot(B.T) + np.diag(eps_var)

    Q = (Q + Q.T) / 2
    # ----------------------------------------------------------------------

    return mu, Q, adj_R2

def FF(returns, factRet):
    """
    Calibrate the Fama-French 3-factor model.

    Returns:
      mu      : n-vector of expected returns
      Q       : n×n asset covariance matrix
      adj_R2  : n-vector of adjusted R² for each asset regression
    """
    # ----------------------------------------------------------------------
    # align dates and drop any rows with missing data
    data = returns.join(factRet[['Mkt_RF','SMB','HML']], how='inner').dropna()
    
    # build design matrix X = [1, Mkt_RF, SMB, HML]
    F = data[['Mkt_RF','SMB','HML']].values    # (T, 3)
    T = F.shape[0]
    X = np.hstack([np.ones((T, 1)), F])        # (T, 4)
    
    assets = returns.columns
    N = len(assets)
    
    # storage
    B       = np.zeros((N, 3))
    alpha   = np.zeros(N)
    eps_var = np.zeros(N)
    adj_R2  = np.zeros(N)
    
    # run OLS for each asset
    for i, asset in enumerate(assets):
        y, *_ = data[asset].values, 
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        
        alpha[i] = coeffs[0]
        B[i, :]  = coeffs[1:]
        
        resid = y - X.dot(coeffs)
        eps_var[i] = resid.var(ddof=1)
        
        # compute R²
        SSR = np.sum(resid**2)
        SST = np.sum((y - y.mean())**2)
        R2  = 1 - SSR/SST
        
        # count only nonzero betas (exclude intercept)
        p_eff = np.count_nonzero(coeffs[1:])
        
        # adjusted R² with p_eff predictors
        adj_R2[i] = 1 - (1 - R2) * (T - 1) / (T - p_eff - 1)
    
    # expected returns
    f_mean = F.mean(axis=0)             # (3,)
    mu     = alpha + B.dot(f_mean)      # (N,)
    
    # factor-model covariance
    Sigma_f = np.cov(F, rowvar=False, ddof=1)  
    Q       = B.dot(Sigma_f).dot(B.T) + np.diag(eps_var)
    # ----------------------------------------------------------------------
    Q = (Q + Q.T) / 2
    return mu, Q, adj_R2


def OLS_with_PCA(returns,
                 factRet,
                 n_components):
    """
    1) Align on dates & drop NaNs
    2) PCA-reduce factRet to `n_components`
    3) Call OLS(returns, pca_factors_df)
    Returns:
      mu, Q  as in original OLS
    """
    # 1) align
    data = returns.join(factRet, how="inner").dropna()
    R = data[returns.columns]
    F = data[factRet.columns]

    # 2) PCA on factors
    pca = PCA(n_components=n_components)
    F_pca = pca.fit_transform(F.values)        # shape (T, k)
    cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(F_pca, index=F.index, columns=cols)

    # 3) call your OLS
    mu, Q, _= OLS(R, df_pca)
    return mu, Q

## test lasso usage, using random data

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_regression
    set_seed = 42
    np.random.seed(set_seed)

    # Generate random data
    n_samples = 1000
    n_features = 8
    n_targets = 5
    # Generate a random regression problem
    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_targets=n_targets, noise=0.1, random_state=set_seed)
    # Convert to DataFrame
    factorReturns = pd.DataFrame(X, columns=['Mkt_RF','SMB','HML','RMW','CMA','Mom','ST_Rev','LT_Rev'])
    periodReturns = pd.DataFrame(y, columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5'])

    # Call the function
    # mu, Q = OLS(periodReturns, factorReturns)
    mu, Q, adj_R2 = LASSO(periodReturns, factorReturns)
    print("Expected Returns (mu):", mu)
    print("Covariance Matrix (Q):", Q)
    print("Adjusted R2:", adj_R2)
    

    