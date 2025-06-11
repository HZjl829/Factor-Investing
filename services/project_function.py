from services.strategies import *


def project_function(periodReturns, periodFactRet, X0=None):
    """
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """

    
    # Strategy = OLS_MVO()
    # Strategy = equal_weight()
    # Strategy = LASSO_MVO()
    # Strategy = RobustBoxMVO()
    
   
    # Strategy = OLS_PCA_MVO()
    # Strategy = EnsembleFactor_MVO()
    Strategy = FactorRiskParity()
    x = Strategy.execute_strategy(periodReturns, periodFactRet)

   
    



    # x = equal_weight(periodReturns)
    return x
