# -*- coding: utf-8 -*-
"""
morrisGSA estimates the elementary elements of the model by using the
method proposed by Morris (1991) and subsequently modified by Campolongo
et al. (2007)

INPUT - n_trj   = number of trajectories to be analyzed
      - Par_info = structure with min and max bounds of the parameters
      - Meas measurement data (if the GSA is performed on an objective
        function (timeseries)
OUTPUT - mu_star= mean of the absolute values for sensitivity indices
         estimated  for each trajectory
       - sigma = standard deviation of the sensitivity indices estimated
        for each trajectory

"""
def morrisGSA(n_trj, par_info, meas_data):
    
    
    ## generate trajectories across the parameter space
    import numpy as np
    from pyDOE import lhs
    
    n_par=len(par_info) #number of parameters    
    n_sim=n_trj*(n_par+1)
    dx=1/n_trj
    
    
    #uses Latin Hypercube Sampling to generate a sample of trajectories 
    # with option that maximize the minimum distance across the points (option cm)
    # note: the combinatory method suggested in Campolongo et al., 2007 is too 
    # computationally heavy, so the "centermaxmin" option of pyDOE is used 
    
    trj_start_norm = lhs(n_par, samples=n_trj, criterion='cm') 
    trj_start = par_info['min']+trj_start_norm*(np.array([par_info['max']-par_info['min']]))
    par_sample=np.empty([0,n_par]) #vector to store all the points of the trajectories
    
    
    for i in range(n_trj):
        trj_point=trj_start[i,:]*(1+np.tril(np.ones([n_par+1,n_par]),-1)*dx)
        par_sample=np.append(par_sample,trj_point,axis=0) 
        
        
    ## run the GSA analysis
    
    obj_fun=np.zeros(n_sim) #vector to store the objective function
    
    import NH4models as fun # import the model to be run
    import objective_functions as objFun # import the objective function to be used

    # run the model for the generated samples
    

    for i in range(n_sim):
        print(['analyzing parameter set ' + str(i) + 'out of ' + str(n_sim)] )
        # run the model
        out = fun.NH4inletModel2(par_sample[i,],meas_data)
        # calculate the objective function
        obj_fun[i]=objFun.MARE(meas_data['smoothed'],out['simNH4load'])
        
    
    ## calculate indices
    d=np.zeros([n_trj,n_par]) #vector to store the objective function
    for i in range(n_trj):
        for j in range(n_par):
            d[i,j]=(obj_fun[i*(n_par+1)+(j+1)]-obj_fun[i*(n_par+1)+j])/obj_fun[i*(n_par+1)+j]/dx
    
    
    mu=np.zeros([n_par])
    mu_star=np.zeros([n_par])
    sigma=np.zeros([n_par])
    for j in range(n_par):
        mu[j]=np.mean(d[:,j])
        sigma[j]=np.std(d[:,j])
        mu_star[j]=np.mean(abs(d[:,j]))
        
    
        
    # make graphs
    import matplotlib.pyplot as plt    
    
    plt.figure()
       
    # bars of mu
    # plt.subplot(2,2,1)
    # x_par = range(n_par)
    # plt.bar(x_par, mu)
    # plt.ylabel(r'$\mu$')
    # plt.xticks(x_par, par_info['name'])

    # bars of mu star
    plt.subplot(2,1,1)
    x_par = range(n_par)
    plt.bar(x_par, mu_star)
    plt.ylabel(r'$\mu^*$')
    plt.xticks(x_par, par_info['name'])        
        
    #  mu vs standard deviation
    # plt.subplot(2,2,3)
    # x_par = range(n_par)
    # plt.plot(mu,sigma,'.')
    # plt.xlabel(r'$\mu$')
    # plt.ylabel(r'$\sigma$')
    # plt.grid(True)
    # #xlim=[np.min([1.05*np.floor(np.min(mu)),0]),np.max([0,1.05*np.ceil(np.max(mu))])]
    # xlim = [-0.25, 1.25]
    # #ylim=[0,np.ceil(np.max(sigma)*1.05)]
    # ylim = [0, 0.1]
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    # for i in range(n_par):
    #     plt.annotate(par_info['name'][i],(mu[i],sigma[i]))    

    #  mu start vs standard deviation
    plt.subplot(2,1,2)
    x_par = range(n_par)
    plt.plot(mu_star,sigma,'.')
    plt.xlabel(r'$\mu^*$')
    plt.ylabel(r'$\sigma$')
    plt.grid(True)
    # xlim=[np.min([1.05*np.floor(np.min(mu_star)),0]),np.max([0,1.05*np.ceil(np.max(mu_star))])]
    xlim = [0, 1.1]
    # ylim=[0,np.ceil(np.max(sigma)*1.05)]
    ylim = [0, 0.1]
    plt.plot([0,np.max([xlim[1],ylim[1]])],[0,np.max([xlim[1],ylim[1]])],color='red' )    
    plt.xlim(xlim)
    plt.ylim(ylim)
    for i in range(n_par):
        plt.annotate(par_info['name'][i],(mu_star[i],sigma[i]))
    

    return  [mu_star,sigma]
    

