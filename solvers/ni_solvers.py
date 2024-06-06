"""
    Purpose: This module implements both deterministic and stochastic variance-reduction methods to solve nolinear inclusion:
                                                    0 in G(x) + T(x).
    where G from R^p to R^p, which is L-Lipschitz continuous and satisfies a weak-Minty solution condition.
        It consists of six differents algorithms.
      1. OGA             - The optimistic gradedient method, e.g., from [1]
      4. VrFRBS          - The forward-reflected-backward splitting algorithm with SVRG variance reduction from [2].
      5. VrEG            - The extragradient algorithm with SVRG variance reduction from [3].
      6. SagaRootFinding - The SAGA-variance reduction root-finding method from [4].

    References:
        [1]. C. Daskalakis, A. Ilyas, V. Syrgkanis, and H.~Zen, Training GANs with Optimism}, in  (ICLR 2018), 2018.
        [2]. A. Alacaoglu, Y. Malitsky, and V. Cevher, Forward-reflected-backward method with variance reduction, Comput. Optim. Appl., 80 (2021).
        [3]. A. Alacaoglu and Y. Malitsky, Stochastic variance reduction for variational inequality methods, arXiv:2102.08352, (2021).
        [4]. D. Davis, Variance reduction for root-finding problems}, Math. Program.,  (2022), pp. 1--36.
"""

import numpy as np
from numpy import linalg as la
import random
import scipy as sci

def OG(data, G_op_eval, J_op_eval, x0, **kwargs):
    """
    OG: An implementation of the optimistic gradient method for solving:
                            0 in G(x) + T(x).
        This method is from [1].
    Args: 
        + data = training data
        + G_op_eval = the function handle to evaluate G(x)
        + J_op_eval = evaluate the resolvant of eta*T
        + x0 = an initial point
        + kwargs = optional and control parameters
    Returns: 
        + opt_sol = approximate solution 
        + message = output message
        + epoch_hist = history.
    """
    
    # parameters
    Lips        = data.get("L")
    gamma       = kwargs.pop('gamma', 0.5)
    eta         = kwargs.pop('eta', 2.0/Lips)
    n_max_iters = kwargs.pop('n_max_iters', 10000)
    tol         = kwargs.pop('tolerance', 1e-8)
    is_term     = kwargs.pop('is_term', True)
    
    # print setup
    verbose    = kwargs.pop('verbose', None)
    print_step = kwargs.pop('print_step', 1)

    # initalization 
    n, p     = data.get("n"), data.get("p")
    full_id  = range(n)
    msg      = "Initialization"
    hist     = []
    
    # initialize the iterate vectors
    x_prev, x_cur = x0.copy(), x0.copy()
    Gx_prev  = G_op_eval(data, x_prev, full_id, n)
    op_norm0 = la.norm((x_prev - J_op_eval(data, x_prev - eta*Gx_prev, eta))/eta)
    
    # print initial information
    if verbose:
        print('Solver: Optimistic Gradient Method for Monotone Inclusions ...')
        print(
            '{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=42, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='Epoch', fill=' ', align='^', width=7, ), '|',
            '{message:{fill}{align}{width}}'.format(message='Error', fill=' ', align='^', width=13, ), '|',
            '{message:{fill}{align}{width}}'.format(message='||G_eta(x)||', fill=' ', align='^', width=13, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='', fill='-', align='^', width=42)
        )

    # main loop -- running up to max_iters iterations.
    for k in range(0, n_max_iters):
        
        # evaluate the operator G(x).
        Gx_cur  = G_op_eval(data, x_cur, full_id, n)
        
        # form the forward-relected operator S(x)
        Sx_cur  = Gx_cur - gamma*Gx_prev

        # update the next iterate
        x_next = J_op_eval( data, x_cur - eta*Sx_cur, eta )

        # compute error and operator norm.
        error   = la.norm(x_next - x_cur)
        op_norm = la.norm((x_cur - J_op_eval(data, x_cur - eta*Gx_cur, eta))/eta)

        # print every print_step iterations.
        if verbose:
            if k % print_step == 0:
                print(
                    '{:^8.0f}'.format(int(k)), '|',
                    '{:^13.3e}'.format(error), '|',
                    '{:^13.3e}'.format(op_norm/op_norm0), '|'
                )
        # save history to plot results.                
        hist.append(dict({"epoch":k, "error": error, "op_norm": op_norm}))

        # termination conditions.
        if is_term and op_norm <= tol*max(op_norm0, 1.0):
            msg = "Convergence acheived!"
            break

        # go to the next iteration.
        x_prev  = x_cur
        x_cur   = x_next
        Gx_prev = Gx_cur
    
    # end of the loop ...
    if verbose:
        print('{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=42, ), '\n')
    if k+1 >= n_max_iters:
        msg = "Exceed the maximum number of epochs. Increase it to run further ..."
        
    return dict({"opt_sol": x_cur, "message": msg, "epoch_hist": hist})
    

def VrFRBS(data, G_op_eval, J_op_eval, x0, mb_size=5, prob=0.05, **kwargs):
    """
    VrFRBS: An Implementation of Variance-Reduced Forward-Reflected Method for solving the following inclusion: 
                                0 in G(x) + T(x).
            This is the algorithm in [2].
    Args: 
        + data = training dataset
        + G_op_eval = the evaluation of operator G(x)
        + J_op_eval = evaluate the resolvant of eta*T
        + mb_size = mini-batch size
        + prob = probability to update snapshot point w.
        + kwargs = optional and control parameters.
    Returns: 
        + opt_sol = approximate solution
        + message = solver messages
        + epoch_hist = history of training process.
    """

    # parameters
    eta      = kwargs.pop('eta', 0.5)
    n_epochs = kwargs.pop('n_epochs', 200)
    tol      = kwargs.pop('tolerance', 1e-8)
    is_term  = kwargs.pop('is_term', True)
    
    # print setup
    verbose    = kwargs.pop('verbose', None)
    print_step = kwargs.pop('print_step', 100)

    # initialization
    n, p    = data.get("n"), data.get("p")
    full_id = range(n)
    msg     = "Initialization"
    n_count = 0
    n_inner_iters  = int(n/mb_size)
    total_iters    = int(n_epochs*n_inner_iters)
    epoch_hist, hist = [], []
    
    # initialize iterate vectors
    w_cur, w_prv, x_cur = x0.copy(), x0.copy(), x0.copy()
    
    # print initial information
    if verbose:
        print('Solver: Variance-Reduced Forward-Reflected-Backward Splitting (Alacaoglu et al 2021) ...')
        print(
            '{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=42, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='Epoch', fill=' ', align='^', width=7, ), '|',
            '{message:{fill}{align}{width}}'.format(message='Error', fill=' ', align='^', width=13, ), '|',
            '{message:{fill}{align}{width}}'.format(message='||G_eta(x)||', fill=' ', align='^', width=13, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='', fill='-', align='^', width=42)
        )

    # evalute the first full operator G(w) at the snapshot point w.
    Gw_cur   = G_op_eval(data, w_cur, full_id, n)
    op_norm  = la.norm((w_cur - J_op_eval(data, w_cur - eta*Gw_cur, eta))/eta)
    op_norm0 = op_norm
    op_norm1 = op_norm
    
    # the main loop to loop over epochs.
    for k in range(total_iters):

        # sample a mini-batch
        mb_id   = random.sample(full_id, mb_size)
        
        # evaluate G at a snap-shot point w_cur.
        Gxi_cur = G_op_eval(data, x_cur, mb_id, mb_size)
        Gwi_prv = G_op_eval(data, w_prv, mb_id, mb_size)
        Sx_cur  = Gw_cur + Gxi_cur - Gwi_prv

        # update the iterate.
        x_next  = J_op_eval(data, x_cur - eta*Sx_cur, eta)

        # compute the error and norm of gradient mapping.
        error   = la.norm(x_next - x_cur)
        op_norm = la.norm((x_cur - J_op_eval(data, x_cur - eta*Sx_cur, eta))/eta)

        # evaluate full operator at snapshot point.
        if np.random.binomial(n=1, p=prob):
            # return to snapshot points
            w_prv    = w_cur
            w_cur    = x_next
            Gw_cur   = G_op_eval(data, w_cur, full_id, n)
            op_norm1 = la.norm((w_cur - J_op_eval(data, w_cur - eta*Gw_cur, eta))/eta)

        # only evaluate G(x) for each epoch.
        if k%n_inner_iters==0:
            epoch_hist.append(dict({"epoch": n_count, "error": error, "op_norm": op_norm1}))
            n_count += 1

        # print every print_step iterations.
        if verbose:
            if k%n_inner_iters==0: #k % print_step == 0:
                print(
                    '{:^8.0f}'.format(int(n_count)), '|',
                    '{:^13.3e}'.format(error), '|',
                    '{:^13.3e}'.format(op_norm/op_norm0), '|'
                )
        # save history to plot results. 
        hist.append(dict({"epoch": n_count, "iter": k, "error": error, "op_norm": op_norm}))

        # checking the termination conditions.
        if is_term and op_norm <= tol*max(op_norm0, 1.0):
            msg = "Convergence acheived!"
            break
            
        # move to the next iteration.
        x_cur  = x_next

    # end of the loop ...
    if verbose:
        print('{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=42, ), '\n')
    if k+1 >= total_iters:
        msg = "Exceed the maximum number of epochs. Increase it to run further ..."
        
    return dict({"opt_sol": x_cur, "message": msg, "hist": hist, "epoch_hist": epoch_hist})
    

def VrEG(data, G_op_eval, J_op_eval, x0, mb_size=5, alpha = 0.5, prob=0.05, **kwargs):
    """
    VrEG: Variance-Reduced Extragradient Method for Solving Monotone Inclusions of the form:
                                0 in G(x) + T(x).
          This is based on the paper of ALACAOGLU and MALITSKY (COLT, 2022).
    Args: 
        + data = training dataset
        + G_op_eval = the evaluation of operator G(x)
        + J_op_eval = evaluate the resolvant of eta*T
        + mb_size = mini-batch size
        + prob = probability to update snapshot point w.
        + kwargs = optional and control parameters.
    Returns: 
        + opt_sol = approximate solution
        + message = solver messages
        + epoch_hist = history of training process.
    """

    # parameters
    eta      = kwargs.pop('eta', 0.5)
    alpha    = kwargs.pop('alpha', 0.5)
    n_epochs = kwargs.pop('n_epochs', 200)
    tol      = kwargs.pop('tolerance', 1e-8)
    is_term  = kwargs.pop('is_term', True)
    
    # print setup
    verbose    = kwargs.pop('verbose', None)
    print_step = kwargs.pop('print_step', 100)

    # initialization
    n, p          = data.get("n"), data.get("p")
    full_id       = range(n)
    msg           = "Initialization"
    n_inner_iters = int(n/mb_size)
    total_iters   = int(n_epochs*n_inner_iters)
    n_count       = 0
    epoch_hist, hist = [], []
    
    # initalize iterate vectors.
    w_cur, x_cur = x0.copy(), x0.copy()
    
    # print initial information
    if verbose:
        print('Solver: Variance-Reduced Extragradient Method (Alacaoglu & Malitsky 2022) ...')
        print(
            '{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=42, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='Epoch', fill=' ', align='^', width=7, ), '|',
            '{message:{fill}{align}{width}}'.format(message='Error', fill=' ', align='^', width=13, ), '|',
            '{message:{fill}{align}{width}}'.format(message='||G_eta(x)||', fill=' ', align='^', width=13, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='', fill='-', align='^', width=42)
        )

    # evalute the first full operator G(w) at the snapshot point w.
    Gw_cur   = G_op_eval(data, w_cur, full_id, n)
    op_norm  = la.norm((w_cur - J_op_eval(data, w_cur - eta*Gw_cur, eta))/eta)
    op_norm0 = op_norm
    op_norm1 = op_norm
    
    # the main loop to loop over epochs.
    for k in range(total_iters):
        
        # update xbar_cur and z_cur
        xbar_cur = (1.0-alpha)*w_cur + alpha*x_cur
        z_cur    = J_op_eval(data, xbar_cur - eta*Gw_cur, eta)

        # sample a mini-batch
        mb_id    = random.sample(full_id, mb_size)
        
        # evaluate G at a snap-shot point w_cur.
        Gzi_cur  = G_op_eval(data, z_cur, mb_id, mb_size)
        Gwi_cur  = G_op_eval(data, w_cur, mb_id, mb_size)
        Sx_cur   = Gw_cur + Gzi_cur - Gwi_cur

        # update the iterate.
        x_next   = J_op_eval(data, xbar_cur - eta*Sx_cur, eta)

        # compute the error and norm of gradient mapping
        error    = la.norm(x_next - x_cur)
        op_norm  = la.norm((x_cur - J_op_eval(data, x_cur - eta*Sx_cur, eta))/eta)

        # evaluate full operator at snapshot point.
        if np.random.binomial(n=1, p=prob):
            # return to snapshot points
            w_cur    = x_next
            Gw_cur   = G_op_eval(data, w_cur, full_id, n)
            op_norm1 = la.norm((w_cur - J_op_eval(data, w_cur - eta*Gw_cur, eta))/eta)

        # only evaluate G(x) for each epoch.
        if k%n_inner_iters==0:
            epoch_hist.append( dict({"epoch": n_count, "error": error, "op_norm": op_norm1}))
            n_count += 1

        # print every print_step iterations.
        if verbose:
            if k%n_inner_iters==0: #k % print_step == 0:
                print(
                    '{:^8.0f}'.format(int(n_count)), '|',
                    '{:^13.3e}'.format(error), '|',
                    '{:^13.3e}'.format(op_norm/op_norm0), '|'
                )
        # save history to plot results. 
        hist.append(dict({"epoch": n_count, "iter": k, "error": error, "op_norm": op_norm}))

        # checking the termination conditions.
        if is_term and op_norm <= tol*max(op_norm0, 1.0):
            msg = "Convergence acheived!"
            break
            
        # move to the next iteration.
        x_cur  = x_next

    # end of the loop ...
    if verbose:
        print('{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=42, ), '\n')
    if k+1 >= total_iters:
        msg = "Exceed the maximum number of epochs. Increase it to run further ..."
        
    return dict({"opt_sol": x_cur, "message": msg, "hist": hist, "epoch_hist": epoch_hist})
    
def SagaRF(data, G_op_eval, Gb_op_eval, J_op_eval, x0, mb_size=5, **kwargs):
    """
    SagaRF: An Implementation of SAGA Variance-Reduced Root-Finding Method for solving the following inclusion:
                                    0 in G(x) + T(x).
            This method from "Variance reduction for root-finding problems" by Davis Damek, Math. Program., 2017.
    Args: 
        + data = training dataset
        + G_op_eval = the evaluation of operator G(x)
        + mb_size = mini-batch size
        + prob = probability to update snapshot point w.
        + kwargs = optional and control parameters.
    Returns: 
        + opt_sol = approximate solution
        + message = solver messages
        + epoch_hist = history of training process.
    """

    # parameters
    Lips     = data.get("L")
    eta      = kwargs.pop('beta', 0.5/Lips)
    rho      = kwargs.pop('rho', 1.0/Lips)
    n_epochs = kwargs.pop('n_epochs', 200)
    tol      = kwargs.pop('tolerance', 1e-8)
    is_term  = kwargs.pop('is_term', True)
    
    # print setup
    verbose    = kwargs.pop('verbose', None)
    print_step = kwargs.pop('print_step', 100)

    # initialization
    n, p          = data.get("n"), data.get("p")
    full_id       = range(n)
    msg           = "Initialization"
    n_inner_iters = int(n/mb_size)
    total_iters   = int(n_epochs*n_inner_iters)
    n_count       = 0
    hist, epoch_hist = [], []
        
    # initalize the iterates 
    x_cur = x0.copy()
    
    # compute y_cur and initialize y_prv
    y_cur = J_op_eval(data, x_cur, rho)
    
    # print initial information
    if verbose:
        print('Solver: SAGA-Root-Finding Algorithm for Monotone Inclusions ...')
        print(
            '{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=42, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='Iters', fill=' ', align='^', width=7, ), '|',
            '{message:{fill}{align}{width}}'.format(message='Error', fill=' ', align='^', width=13, ), '|',
            '{message:{fill}{align}{width}}'.format(message='||G(x)||', fill=' ', align='^', width=13, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='', fill='-', align='^', width=42)
        )

    # evalute the first full operator G(w) at the snapshot point w.
    Gy_cur, Gop_memory  = Gb_op_eval(data, y_cur, full_id, n)
    op_norm  = la.norm( Gy_cur + (x_cur - y_cur)/rho )
    op_norm0 = op_norm
    error    = 0.0

    # initialize a table to store history of Gi(x)
    Gy_avg = np.mean(Gop_memory, axis=0)  
    
    # Add the first epoch to the epoch history.
    epoch_hist.append(dict({"epoch": n_count, "error": error, "op_norm": op_norm}))
    
    # the main loop to loop over epochs.
    for k in range(total_iters):

        # sample a mini-batch
        mb_id   = random.sample(full_id, mb_size)
        
        # evaluate Gy(k) and Gy(k-1) for a given mini-batch.
        y_cur   = J_op_eval(data, x_cur, rho)
        Gyi_cur, Gyi_mb = Gb_op_eval(data, y_cur, mb_id, mb_size)

        # compute Gz(k) for full data and Gz(k) for the mini-batch.
        Gzi_avg = np.mean([Gop_memory[i] for i in mb_id], axis=0)
        Sy_cur  = Gy_avg - Gzi_avg + Gyi_cur
        Sx_cur  = Sy_cur + (x_cur - y_cur)/rho
        
        # store G(x) into a table.
        for i_e, i in enumerate(mb_id):
            Gop_memory[i] = Gyi_mb[i_e]
        Gy_avg   = np.mean(Gop_memory, axis=0)
        op_norm1 = la.norm(Gy_avg + (x_cur - y_cur)/rho)
        
        # update the next iterate
        x_next  = x_cur  - eta*Sx_cur

        # compute the error and operator norm.
        error   = la.norm(x_next - x_cur)
        op_norm = la.norm( Sx_cur )
        
        # only evaluate G(x) for each epoch.
        if k%n_inner_iters==0:
            n_count += 1
            epoch_hist.append( dict({"epoch": n_count, "error": error, "op_norm": op_norm}))
            
        # print every print_step iterations.
        if verbose:
            if k%n_inner_iters==0: #k % print_step == 0:
                print(
                    '{:^8.0f}'.format(int(n_count)), '|',
                    '{:^13.3e}'.format(error), '|',
                    '{:^13.3e}'.format(op_norm/op_norm0), '|'
                )
        # save history to plot results. 
        hist.append(dict({"iter":k, "error": error, "op_norm": op_norm}))

        # checking the termination conditions.
        if is_term and op_norm <= tol*max(op_norm0, 1.0):
            msg = "Convergence acheived!"
            break
            
        # move to the next iteration.
        x_cur  = x_next

    # end of the loop ...
    if verbose:
        print('{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=42, ), '\n')
    if k+1 >= total_iters:
        msg = "Exceed the maximum number of epochs. Increase it to run further ..."
    return dict({"opt_sol": x_cur, "message": msg, "hist": hist, "epoch_hist": epoch_hist})
