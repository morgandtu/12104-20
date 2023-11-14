# -*- coding: utf-8 -*-
"""
GSAregression estimates the sensitivity indices by using the regression
method listed in Saltelli and Annoni (2010)

INPUT - Par_inf: structure with min and max bounds of the parameters
       - Meas measurement data (if the GSA is performed on an objective
         function (timeseries)
       - n_sim: number of samples to be analyzed
 OUTPUT - sensitivity indices for each parameter

OBS this function requires the following packages 
    - pyDOE 
    
(if not available, packages need to be installed befor you run the code for 
the first time - see https://conda.io/docs/user-guide/tasks/manage-pkgs.html)  

Version 20180821  - LUVE

"""

def GSAregression(par_info, meas_data, sim_time_step, n_sim):
    
    import numpy as np
    import statsmodels.api as sm

    
    ## generate sample of points to be analyzed across the parameter space
    from pyDOE import lhs

    n_par=len(par_info) #number of parameters
    
    par_sample_norm = lhs(n_par, samples=n_sim, criterion='center') #uses Latin Hypercube Sampling to generate a sample of parameters   
    par_sample= par_info['min']+par_sample_norm*(np.array([par_info['max']-par_info['min']]))
    
    ## run the GSA analysis
    
    # run the GSA analysis
    obj_fun = np.zeros(n_sim) #vector to store the objective function
    import degradation_function as fun # import the model to be run
    import objective_functions as objFun # import the objective function to be used

    # run the model for the generated samples
    for i in range(n_sim):
        # print progress status
        print(['analyzing parameter set ' + str(i) + ' out of ' + str(n_sim)] )
        # run the model
        out = fun.degradation_rate(sim_time_step,par_sample[i,])
        # calculate the objective function
        obj_fun[i] = objFun.invMSE(meas_data,out)
    
    # calculate the regression indices
    # (use normalized values, otherwise the biggest parameter in abs term will 
    # most sensitive)

    xSample= sm.add_constant(par_sample_norm) #add a constant (so intercept is considere in regression)
    regr_results = sm.OLS(obj_fun,xSample).fit()
    

    sRegr=abs(regr_results.params[1:n_par+1]) # first value is neglected, since it refer to the intercept
    
    # make graphs
    import matplotlib.pyplot as plt    
    
    # plot indices
    plt.figure()
    x_par = range(n_par)
    plt.bar(x_par, sRegr)
    plt.xticks(x_par, par_info['name'])
    plt.grid(True)
    plt.title('indices from regression analysis')
    plt.show()


    # pplot parameters against objFun for visual inspection of sensitivity
    plt.figure()
    n_fig = int(np.ceil(n_par**.5))
    
    for i in range(n_par):
        plt.subplot(n_fig,n_fig,i+1)
        plt.plot(par_sample[:,i], obj_fun, '.')
        plt.ylabel('obj fun')
        plt.xlabel(par_info['name'][i])
        plt.grid(True)


    return  sRegr