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

    #### Graphical Comparison

def dynamic_graph():

    w_pinotti = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6244443, 0.3755557, 0]).reshape(15, 1)
    w_becker = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4303541, 0.4893414, 0.0803045]).reshape(15,1)

    y_synth_pinotti = w_pinotti.T @ y_control_all
    y_synth_becker = w_becker.T @ y_control_all
    y_synth_basic = w_basic.T @ y_control_all

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_basic[0],
                    mode='lines', name='Optimizer'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_pinotti[0],
                    mode='lines', name='Pinotti'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_becker[0],
                    mode='lines', name='Becker and Klößner'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_treat_all[0],
                    mode='lines', name='Treated unit'))

    fig.add_shape(dict(type="line", x0=1960, y0=0, x1=1960, y1=11000,
                   line=dict(color="Black", width=1)))

    fig.add_shape(dict(type="line", x0=1974, y0=0, x1=1974, y1=11000,
                   line=dict(color="Black", width=1)))

    fig.add_shape(dict(type="line", x0=1980, y0=0, x1=1980, y1=11000,
                   line=dict(color="Black", width=1)))

    fig.add_trace(go.Scatter(x=[1960], y=[12000], mode="text",
         name="Matching", text=["End of Matching<br>Period"]))
  
    fig.add_trace(go.Scatter(x=[1974], y=[12000], mode="text",
         name="Event 1", text=["Drug<br>Smuggling"]))

    fig.add_trace(go.Scatter(x=[1981], y=[12000], mode="text",
         name="Event 2", text=["Basilicata<br>Earthquake"]))

    fig.update_layout(title='Figure 3.1: Synthetic Control Optimizer vs. Treated unit',
                   xaxis_title='Time', yaxis_title='GDP per Capita')

    # Dynamic graph
    fig.show()
    
    
 def RMSPE_compare1():
    # Function to obtain Root Mean Squared Prediction Error 
    def RMSPE(w):
        return np.sqrt(np.mean((Z1 - Z0 @ w)**2))
    
# Dataframe to compare RMSPE values
    RMSPE_values = [RMSPE(w_basic), RMSPE(w_pinotti), RMSPE(w_becker)]
    method = ['RMSPE CVXPY','RMSPE Pinotti','RMSPE Becker']
    RMSPE_compare = pd.DataFrame({'Outcome':RMSPE_values}, index=method)
    display(RMSPE_compare)
    
    
# Dataframe to show predicted vs actual values of variables.
def data_compare():
    
    x_pred_pinotti = (X0 @ w_pinotti)
    x_pred_basic = (X0 @ w_basic)

    pred_error_pinotti = x_pred_pinotti - X1
    pred_error_basic = x_pred_basic - X1

    data_compare = pd.DataFrame({'Observed':X1.T[0],
                             'Pinotti Predicted':x_pred_pinotti.T[0],
                             'Optimizer Predicted':x_pred_basic.T[0],
                             'Pinotti Differential': pred_error_pinotti.T[0],
                             'Optimizer Differential': pred_error_basic.T[0]},
                              index= data.columns[[3,16,11,12,13,14,26,28]])

#print ('\nBreakdown across predictors:')

    display(data_compare)

#print('\nRMSPE CVXPY: {} \nRMSPE Pinotti: {} \nRMSPE Becker: {}'\
#      .format(RMSPE(w_basic),RMSPE(w_pinotti),RMSPE(w_becker)))


def best_weights():


    n = 100000            # Number of iterations: set to 100000

    iteration_2 = []

# Function to run in parallel 
    ## use Parallel() to save time
    ## n_jobs=-1 -> all CPU used
    ## delayed(f)(x) for x in list(range(1,n+1))  -> computes function f in parallel, for var x from 1 to n+1
    def f(x):
    
        np.random.seed(x)
        v_diag  = np.random.dirichlet(np.ones(8), size=1)
        w_cvxpy = w_optimize(v_diag)[0]
        print(w_cvxpy.shape)
        prediction_error =  RMSPE(w_cvxpy) 
        output_vec = [prediction_error, v_diag, w_cvxpy]

        return output_vec
    
    iteration_2 = Parallel(n_jobs=-1)(delayed(f)(x) for x in list(range(1,n+1)))

# Organize output into dataframe
    solution_frame_2 = pd.DataFrame(iteration_2)
    solution_frame_2.columns =['Error', 'Relative Importance', 'Weights']

    solution_frame_2 = solution_frame_2.sort_values(by='Error', ascending=True)

    w_cvxpy = solution_frame_2.iloc[0][2]
    v_cvxpy = solution_frame_2.iloc[0][1][0]

    best_weights_region = pd.DataFrame({'Region':control_units.region.unique(),
                                    'W(V*)': np.round(w_cvxpy.ravel(), decimals=3)})

    best_weights_importance = pd.DataFrame({'Predictors': data.columns[[3,16,11,12,13,14,26,28]],
                                        'V*': np.round(v_cvxpy, 3)})

    display(best_weights_region)
    display(best_weights_importance)
#display(best_weights_importance)
#display(best_weights_region)
 
def RMSPE_compare2():
    A = X0
    b = X1.ravel()  ## .ravel() returns continuous flattened array [[a,b],[c,d]]->[a,b,c,d]
    iteration_3 = []
    init_w = [0]*15

    bnds = ((0, 1),)*15
    cons = ({'type': 'eq', 'fun': lambda x: 1.0 -  np.sum(x) })   ## constraint

    def fmin(x,A,b,v):         ## function we want to min
        c = np.dot(A, x) - b   ## np.dot(a,b) multiplies a and b => X0*x - X1
        d = c ** 2
        y = np.multiply(v,d)   ## y = v * (X0*x - X1)^2
        return np.sum(y)

    def g(x):
    
        np.random.seed(x)    ## deterministic random number generation by setting seed
        v = np.random.dirichlet(np.ones(8), size=1).T
        args = (A,b,v)
        res = optimize.minimize(fmin,init_w,args,method='SLSQP',bounds=bnds,
                        constraints=cons,tol=1e-10,options={'disp': False})
        ## optimize.minimize(objective, initial guess, arguments, 'SLSPQ'=sequential least squares programming,
        ##                   bounds, constraints, tolerance, )
    
        prediction_error =  RMSPE(res.x) 
        output_vec = [prediction_error, v, res.x]

        return output_vec
    
    iteration_3 = Parallel(n_jobs=-1)(delayed(g)(x) for x in list(range(1,n+1)))
    # Organize output into dataframe
    solution_frame_3 = pd.DataFrame(iteration_3)
    solution_frame_3.columns =['Error', 'Relative Importance', 'Weights']

    solution_frame_3 = solution_frame_3.sort_values(by='Error', ascending=True)

    w_scipy = solution_frame_3.iloc[0][2].reshape(15,1)
    v_scipy = solution_frame_3.iloc[0][1].T[0]

    best_weights_region2 = pd.DataFrame({'Region':control_units.region.unique(), 
                                    'W(V*)': np.round(w_scipy.ravel(), decimals=3)})

    best_weights_importance2 = pd.DataFrame({'Predictors': data.columns[[3,16,11,12,13,14,26,28]],
                                        'V*': np.round(v_scipy, 3)})

    display(best_weights_importance2)
    display(best_weights_region2)


    y_synth_scipy = w_scipy.T @ y_control_all
    y_synth_cvxpy = w_cvxpy.T @ y_control_all

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_cvxpy[0],
                    mode='lines', name='Optimizer CVXPY'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_scipy[0],
                    mode='lines', name='Optimizer SciPY'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_pinotti[0],
                    mode='lines', name='Pinotti'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_treat_all[0],
                    mode='lines', name='Treated unit'))
    fig.add_shape(dict(type="line", x0=1960, y0=0, x1=1960, y1=11000,
                   line=dict(color="Black", width=1)))

    fig.add_trace(go.Scatter(x=[1960], y=[12000], mode="text",
        name="Matching", text=["End of Matching<br>Period"]))

    fig.update_layout(title='Figure 3.2: Synthetic Control Optimizer vs. Treated unit',
                   xaxis_title='Time', yaxis_title='GDP per Capita')
    fig.show()

    RMSPE_values2 = [RMSPE(w_cvxpy), RMSPE(w_scipy), RMSPE(w_pinotti)]
    method2 = ['RMSPE CVXPY','RMSPE scipy','RMSPE Pinotti']
    RMSPE_compare2 = pd.DataFrame({'RMSE':RMSPE_values2}, index=method2)
    display(RMSPE_compare2)

#print('\nRMSPE CVXPY: {} \nRMSPE scipy: {} \nRMSPE Pinotti: {}'\
#      .format(RMSPE(w_cvxpy),RMSPE(w_scipy),RMSPE(w_pinotti)))


def data_prep(data,unit_identifier,time_identifier,matching_period,treat_unit,control_units,outcome_variable,
              predictor_variables):
    
    X = data.loc[data[time_identifier].isin(matching_period)]
    X.index = X.loc[:,unit_identifier]
    
    X0 = X.loc[(X.index.isin(control_units)),(predictor_variables)] 
    X0 = X0.groupby(X0.index).mean().values.T
    
    X1 = X.loc[(X.index == treat_unit),(predictor_variables)]
    X1 = X1.groupby(X1.index).mean().values.T

    Z0 = np.array(X.loc[(X.index.isin(control_units)),(outcome_variable)]).reshape(len(control_units),len(matching_period)).T
    Z1 = np.array(X.loc[(X.index == treat_unit),(outcome_variable)]).reshape(len(matching_period),1)
    
    return X0, X1, Z0, Z1

# Function to avoid re-writing code during sensitivity analysis.
def SCM(data,unit_identifier,time_identifier,matching_period,treat_unit,control_units,outcome_variable,
              predictor_variables,reps = 1):
    
       
       
    def w_optimize(v):

        W = cp.Variable((len(control_units), 1), nonneg=True)
        objective_function    = cp.Minimize(cp.norm(cp.multiply(v, X1 - X0 @ W)))
        objective_constraints = [cp.sum(W) == 1]
        objective_solution    = cp.Problem(objective_function, objective_constraints).solve(verbose=False)
        return (W.value)
    
    def vmin(v):

        v = v.reshape(len(predictor_variables),1)
        W = w_optimize(v)
        return ((Z1 - Z0 @ W).T @ (Z1 - Z0 @ W)).ravel()

    def constr_f(v):
        return float(np.sum(v))

    def constr_hess(x,v):
        v=len(predictor_variables)
        return np.zeros([v,v])

    def constr_jac(v):
        v=len(predictor_variables)
        return np.ones(v)

    def RMSPE_f(w):
        return np.sqrt(np.mean((w.T @ Z0.T - Z1.T)**2))
    
    def v_optimize(i):
    
        bounds  = [(0,1)]*len(predictor_variables)
        nlc     = NonlinearConstraint(constr_f, 1, 1, constr_jac, constr_hess)
        result  = differential_evolution(vmin, bounds, constraints=(nlc),seed=i,tol=0.01)
        v_estim = result.x.reshape(len(predictor_variables),1)  
        return (v_estim)
    
    def h(x):
    
        v_estim1 = v_optimize(x)
        w_estim1 = w_optimize(v_estim1)
        prediction_error = RMSPE_f(w_estim1)
        output_vec = [prediction_error, v_estim1, w_estim1]
        return output_vec

    iterations = []
    iterations = Parallel(n_jobs=-1)(delayed(h)(x) for x in list(range(1,reps+1)))
      
    solution_frame = pd.DataFrame(iterations)
    solution_frame.columns =['Error', 'Relative Importance', 'Weights']
    solution_frame = solution_frame.sort_values(by='Error', ascending=True)

    w_nested = solution_frame.iloc[0][2]
    v_nested = solution_frame.iloc[0][1].T[0]
    
    output = [solution_frame,w_nested,v_nested,RMSPE_f(w_nested)]
    
    return output

def nested():
  
    reps = 1
    treat_unit = 21
    unit_identifier  = 'reg'
    time_identifier  = 'year'
    matching_period  = list(range(1951, 1961))
    control_units    = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20]
    outcome_variable = ['gdppercap']
    entire_period    = list(range(1951, 2008))
    predictor_variables = ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'density']

    data_preparation(data,unit_identifier,time_identifier,matching_period,treat_unit,control_units,outcome_variable,
              predictor_variables)
    
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,treat_unit,
                    control_units,outcome_variable,predictor_variables,reps)

    # Organize output into dataframe
    solution_frame_4 = output_object[0]
    w_nested = output_object[1]
    v_nested = output_object[2]
    control_units = data[(data.reg <= 14) | (data.reg ==20)]

    best_weights_region3 = pd.DataFrame({'Region':control_units.region.unique(), 
                                    'W(V*)': np.round(w_nested.ravel(), decimals=3)})

    best_weights_importance3 = pd.DataFrame({'Predictors': data.columns[[3,16,11,12,13,14,26,28]],
                                        'V*': np.round(v_nested, 3)})

    #display(best_weights_importance3)
    #display(best_weights_region3)

    print('\nOptimizer Weights: {} \nPaper Weights:  {}'\
      .format(np.round(w_nested.T,3), np.round(w_pinotti,3).T))


    print('\nRMSPE Nested:    {} \nRMSPE Pinotti:   {}'\
      .format(np.round(RMSPE(w_nested),5), np.round(RMSPE(w_pinotti),5)))

def unrestricted():
    
    W = cp.Variable((15, 1), nonneg=True)
    objective_function    = cp.Minimize(np.mean(cp.norm(Z1 - Z0 @ W)))
    objective_constraints = [cp.sum(W) == 1]
    objective_solution    = cp.Problem(objective_function, objective_constraints).solve(verbose=False)

    V = cp.Variable((8, 1), nonneg=True)
    objective_function    = cp.Minimize(cp.norm(cp.multiply(V, X1 - X0 @ W.value)))
    objective_constraints = [cp.sum(V) == 1]
    objective_solution    = cp.Problem(objective_function, objective_constraints).solve(verbose=False)

    v_global = V.value.ravel()
    w_global = W.value

    print('\nOptimizer Weights: {} \nOptimal Weights:  {}'\
      .format(np.round(w_global.T,5), np.round(w_becker,5).T))

    print('\nRMSPE Global:   {} \nRMSPE Becker:    {}'\
      .format(np.round(RMSPE(w_global),6), np.round(RMSPE(w_becker),6)))
