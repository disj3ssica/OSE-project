def data_prep():
    
    # Specify conditions for treated unit and control units as per Pinotti's paper (c.f. F216), 
    # 21 is "NEW" Recent mafia presence: Apulia and Basilicata

    treat_unit     = data[data.reg == 21]
    treat_unit     = treat_unit[treat_unit.year <= 1960]                 # Matching period: 1951 to 1960
    treat_unit_all = data[data.reg == 21]                                # Entire period:   1951 to 2007

    control_units     = data[(data.reg <= 14) | (data.reg ==20)]
    control_units     = control_units[control_units.year <= 1960]
    control_units_all = data[(data.reg <= 14) | (data.reg ==20)]

    # Extract the outcome variable for treatment and control unit, y: GDP per capita

    y_treat     = np.array(treat_unit.gdppercap).reshape(1, 10)              # Matching period: 1951 to 1960
    y_treat_all = np.array(treat_unit_all.gdppercap).reshape(1, 57)          # Entire period:   1951 to 2007

    y_control     = np.array(control_units.gdppercap).reshape(15, 10)
    y_control_all = np.array(control_units_all.gdppercap).reshape(15, 57)

    Z1 = y_treat.T      # Transpose
    Z0 = y_control.T

    ## Prepare matrices with only the relevant variables into CVXPY format, predictors k = 8
    predictor_variables = ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'density']
    X = data.loc[data['year'].isin(list(range(1951, 1961)))]
    X.index = X.loc[:,'reg']

    # k x J matrix: mean values of k predictors for J untreated units
    X0 = X.loc[(X.index <= 14) | (X.index ==20),(predictor_variables)] 
    X0 = X0.groupby(X0.index).mean().values.T

    # k x 1 vector: mean values of k predictors for 1 treated unit
    X1 = X.loc[(X.index == 21),(predictor_variables)]
    X1 = X1.groupby(X1.index).mean().values.T
    
    return (X0,X1)
  
  # CVXPY Setup: Define function to call and output a vector of weights function


def cvxpy_solution():
    
    data_prep()
    
    def w_optimize(v=None):
    
        V = np.zeros(shape=(8, 8))
        if v is None:
            np.fill_diagonal(V, [1/8]*8)
        else:
            np.fill_diagonal(V, v)
            
        X0,X1 = data_prep()
        W = cp.Variable((15, 1), nonneg=True) ## Creates a 15x1 positive nonnegative variable
        objective_function    = cp.Minimize(cp.sum(V @ cp.square(X1 - X0 @ W)))
        objective_constraints = [cp.sum(W) == 1]
        objective_solution    = cp.Problem(objective_function, objective_constraints).solve(verbose=False)
    
        return (W.value,objective_solution)


    # CVXPY Solution
    w_basic, objective_solution = w_optimize()
    print('\nObjective Value: ', objective_solution)
    #print('\nObjective Value: ', objective_solution, '\n\nOptimal Weights: ', w_basic.T)
    solution_frame_1 = pd.DataFrame({'Region':control_units.region.unique(), 
                           'Weights': np.round(w_basic.T[0], decimals=3)})

    display(solution_frame_1)
