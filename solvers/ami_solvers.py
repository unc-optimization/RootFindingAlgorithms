"""
    Purpose: This module implements both accelerated deterministic and stochastic accelerated variance-reduction methods 
            to solve monotone inclusions:
                                                    0 in G(x) + T(x).
    Here, G from R^p to R^p, which is L-Lipschitz continuous and monotone, and 
          T from R^p to 2^{R^p} is a maximally monotone operator.
    It consists of six differents algorithms.
      1. AOG       - The accelerated optimistic gradedient method in [1].
      2. SvrgAcFR  - The accelerated forward-reflected algorithm with SVRG variance reduction.
      3. SagaAcFR  - The accelerated forward-reflected algorithm with SAGA variance reduction.
      
    References:
        [1]. Q. Tran-Dinh, From {H}alpern's fixed-point iterations to Nesterov's accelerated interpretations 
             for root-finding problems, Comput. Optim. Appl., 87 (2024), pp. 181--218.
"""

import numpy as np
from numpy import linalg as la
import random
import scipy as sci

def SVRG(data, G_opr, Gw_cur, w_cur, x_cur, x_prv, gamma, mb_id, mb_size):
    """
    Purpose: Compute an SVRG estimator for the forward-reflected operator S.
    Args: 
        + data, G_opr=the operator
        + w_cur=snapshot point, x_cur=x(k), x_prv=x(k-1)
        + gamma = the parameter of forward-reflected operator
        + mb_id = the indices of mini-batch, mb_size = size of mini-batch.
    Returns: 
        + S_k = SVRG( G(w), Gb(w), Gb(x(k), Gb(x(k-1))
    """
    Gwi_cur = G_opr(data, w_cur, mb_id, mb_size)
    Gxi_cur = G_opr(data, x_cur, mb_id, mb_size)
    Gxi_prv = G_opr(data, x_prv, mb_id, mb_size)
    return (1-gamma)*(Gw_cur - Gwi_cur) + Gxi_cur - gamma*Gxi_prv

def AOG(data, G_op_eval, J_op_eval, x0, **kwargs):
    """
    AOG: An implementation of the optimistic gradient method for solving 0 in G(x) + T(x).
    Args:
        + data = training data
        + G_op_eval = the function handle to evaluate G(x)
        + x0 = an initial point
        + kwargs = optional and control parameters
    Returns: 
        + x_cur = approximate solution
        + msg = output message
        + epoch_hist = history.
    """
    
    # parameters
    Lips        = data.get("L")
    beta        = kwargs.pop('beta', 2.0/Lips)
    rho         = kwargs.pop('rho', 1.0/Lips)
    r_shift     = kwargs.pop('r_shift', 1.0)
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
    
    # initialize the iterate vectors
    x_prev, x_cur = x0.copy(), x0.copy()

    # evaluate operators
    y_cur    = J_op_eval(data, x_cur, rho)
    y_prev   = y_cur
    Gy_prev  = G_op_eval(data, y_prev, full_id, n)
    op_norm0 = la.norm( Gy_prev + (x_prev - y_prev)/rho )
    error    = 0.0
        
    # print initial information
    if verbose:
        print('Solver: Accelerated Optimistic Gradient Method for Monotone Inclusions ...')
        print(
            '{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=42, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='Epoch', fill=' ', align='^', width=7, ), '|',
            '{message:{fill}{align}{width}}'.format(message='Error', fill=' ', align='^', width=13, ), '|',
            '{message:{fill}{align}{width}}'.format(message='||G(x)||', fill=' ', align='^', width=13, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='', fill='-', align='^', width=42)
        )

    # Add the first epoch to the epoch history.
    epoch_hist = [dict({"epoch": 0, "error": error, "op_norm": op_norm0})]
    
    # main loop -- running up to max_iters iterations.
    for k in range(0, n_max_iters):
        
        # update the parameters
        theta_cur = (k+1)/(k+r_shift+3)
        gamma_cur = (k+1)/(k+r_shift+1)
        eta_cur   = 2.0*beta*(k+r_shift+1)/(k+r_shift+3)

        # evaluate the operator G(x).
        y_cur   = J_op_eval(data, x_cur, rho)
        Gy_cur  = G_op_eval(data, y_cur, full_id, n)
        
        # form the forward-relected operator S(x)
        Sx_cur  = Gy_cur - gamma_cur*Gy_prev + (x_cur - y_cur)/rho - gamma_cur*(x_prev - y_prev)/rho

        # update the next iterate
        x_next  = x_cur + theta_cur*(x_cur - x_prev) - eta_cur*Sx_cur

        # compute error and operator norm.
        error   = la.norm(x_next - x_cur)
        op_norm = la.norm(Gy_cur + (x_cur - y_cur)/rho)

        # print every print_step iterations.
        if verbose:
            if k % print_step == 0:
                print(
                    '{:^8.0f}'.format(int(k)), '|',
                    '{:^13.3e}'.format(error), '|',
                    '{:^13.3e}'.format(op_norm/op_norm0), '|'
                )
        # save history to plot results.                
        epoch_hist.append(dict({"epoch":k+1, "error": error, "op_norm": op_norm}))

        # termination conditions.
        if is_term and op_norm <= tol*max(op_norm0, 1.0):
            msg = "Convergence acheived!"
            break

        # go to the next iteration.
        x_prev  = x_cur
        y_prev  = y_cur
        x_cur   = x_next
        Gy_prev = Gy_cur
    
    # end of the loop ...
    if verbose:
        print('{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=42, ), '\n')
    if k+1 >= n_max_iters:
        msg = "Exceed the maximum number of epochs. Increase it to run further ..."
        
    return dict({"opt_sol": x_cur, "message": msg, "epoch_hist": epoch_hist})
    
def SvrgAcFR(data, G_op_eval, J_op_eval, x0, mb_size=5, prob=0.05, **kwargs):
    """
    SvrgAcFR: An Implementation of Accelerated Variance-Reduced Forward-Reflected Method for solving 0 in G(x) + T(x).
            This is a single-loop implememtation (known as loopless SVRG).
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
    beta     = kwargs.pop('beta', 0.5/Lips)
    rho      = kwargs.pop('rho', 1.0/Lips)
    r_shift  = kwargs.pop('r_shift', 1.0)
    n_epochs = kwargs.pop('n_epochs', 200)
    tol      = kwargs.pop('tolerance', 1e-8)
    is_term  = kwargs.pop('is_term', True)
    
    # print setup
    verbose    = kwargs.pop('verbose', None)
    print_step = kwargs.pop('print_step', 100)

    # initialization
    n, p    = data.get("n"), data.get("p")
    full_id = range(n)
    n_count = 0
    msg     = "Initialization"
    epoch_hist, hist = [], []
    n_inner_iters = int(n/mb_size)
    total_iters   = int(n_epochs*n_inner_iters)
    gamma_cur     = 1/(r_shift+1)

    # initialize the iterate vectors
    x_prv, x_cur = x0.copy(), x0.copy()
    
    # compute y_cur and initialize y_prv
    y_cur = J_op_eval(data, x_cur, rho)
    y_prv = y_cur.copy()
    w_cur = y_cur.copy()
    
    # print initial information
    if verbose:
        print('Solver: Accelerated Loopless-SVRG-Forward-Reflected Method for Monotone Inclusions ...')
        print(
            '{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=42, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='Epoch', fill=' ', align='^', width=7, ), '|',
            '{message:{fill}{align}{width}}'.format(message='Error', fill=' ', align='^', width=13, ), '|',
            '{message:{fill}{align}{width}}'.format(message='||G(x)||', fill=' ', align='^', width=13, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='', fill='-', align='^', width=42)
        )

    # evalute the first full operator G(w) at the snapshot point w.
    Gw_cur   = G_op_eval(data, w_cur, full_id, n)
    op_norm  = la.norm( Gw_cur + (x_cur - y_cur)/rho )
    op_norm0 = op_norm
    op_norm1 = op_norm
    error    = 0.0

    # Add the first epoch to the epoch history.
    epoch_hist.append(dict({"epoch": n_count, "error": error, "op_norm": op_norm}))
    
    # the main loop to loop over epochs.
    for k in range(total_iters):

        # update the parameters
        theta_cur = (k+1)/(k+r_shift+3)
        gamma_cur = (k+1)/(k+r_shift+1)
        eta_cur   = 2.0*beta*(k+r_shift+1)/(k+r_shift+3)

        # sample a mini-batch
        mb_id   = random.sample(full_id, mb_size)
        
        # evaluate G at a snap-shot point w_cur.
        y_cur   = J_op_eval(data, x_cur, rho)
        Sy_cur  = SVRG(data, G_op_eval, Gw_cur, w_cur, y_cur, y_prv, gamma_cur, mb_id, mb_size)
        Sx_cur  = Sy_cur + (x_cur - y_cur)/rho - gamma_cur*(x_prv - y_prv)/rho
        
        # update the iterate.
        x_next  = x_cur + theta_cur*(x_cur - x_prv) - eta_cur*Sx_cur

        # compute the error and norm of operator
        error   = la.norm(x_next - x_cur)
        op_norm = la.norm( Sx_cur )
        
        # evaluate full operator at snapshot point.
        if np.random.binomial(n=1, p=prob):
            # return to snapshot points
            y_next   = J_op_eval(data, x_next, rho)
            w_cur    = y_next
            Gw_prv   = Gw_cur
            Gw_cur   = G_op_eval(data, w_cur, full_id, n)
            op_norm1 = la.norm( Gw_cur + (x_next - y_next)/rho )

        # only evaluate G(x) for each epoch.
        if k%n_inner_iters==0:
            n_count += 1
            #epoch_hist.append(dict({"epoch": n_count, "error": error, "op_norm": op_norm1}))
            epoch_hist.append(dict({"epoch": n_count, "error": error, "op_norm": op_norm}))
            
        # print every print_step iterations.
        if verbose:
            if k%n_inner_iters==0: #k % print_step == 0:
                print(
                    '{:^8.0f}'.format(int(n_count)), '|',
                    '{:^13.3e}'.format(error), '|',
                    '{:^13.3e}'.format(op_norm/op_norm0), '|'
                )
        # save history to plot results. 
        hist.append(dict({"iter":k, "epoch":n_count, "error": error, "op_norm": op_norm}))

        # checking the termination conditions.
        if is_term and op_norm <= tol*max(op_norm0, 1.0):
            msg = "Convergence acheived!"
            break
            
        # move to the next iteration.
        x_prv  = x_cur
        y_prv  = y_cur
        x_cur  = x_next
        
    # end of the loop ...
    if verbose:
        print('{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=42, ), '\n')
    if k+1 >= total_iters:
        msg = "Exceed the maximum number of epochs. Increase it to run further ..."

    return dict({"opt_sol": x_cur, "message": msg, "hist": hist, "epoch_hist": epoch_hist})

def SagaAcFR(data, G_op_eval, Gb_op_eval, J_op_eval, x0, mb_size=5, **kwargs):
    """
    SagaAcFR: An Implementation of SAGA Accelerated Variance-Reduced Forward-Reflected Method for solving G(x) = 0.
              This is a single-loop implememtation of SAGA-FR method.
    Args: 
        + data = training dataset
        + G_op_eval = the evaluation of operator G(x)
        + Gb_op_eval = the evaluation of operator G(x) and return component Gi(x)
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
    beta     = kwargs.pop('beta', 0.5/Lips)
    rho      = kwargs.pop('rho', 1.0/Lips)
    r_shift  = kwargs.pop('r_shift', 1.0)
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
    gamma_cur     = 1.0/(r_shift + 1.0)
    hist, epoch_hist = [], []
        
    # initalize the iterates 
    x_prv, x_cur = x0.copy(), x0.copy()
    
    # compute y_cur and initialize y_prv
    y_cur = J_op_eval(data, x_cur, rho)
    y_prv = y_cur.copy()
    
    # print initial information
    if verbose:
        print('Solver: Accelerated SAGA-Forward-Reflected Method for Monotone Inclusions ...')
        print(
            '{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=42, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='Iters', fill=' ', align='^', width=7, ), '|',
            '{message:{fill}{align}{width}}'.format(message='Error', fill=' ', align='^', width=13, ), '|',
            '{message:{fill}{align}{width}}'.format(message='||G(x)||', fill=' ', align='^', width=13, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='', fill='-', align='^', width=42)
        )

    # evalute the first full operator G(w) at the snapshot point w.
    Gy_cur, Gop_memory  = Gb_op_eval(data, y_cur, full_id, n)
    Gy_prv = Gy_cur.copy()
    op_norm  = la.norm( Gy_cur + (x_cur - y_cur)/rho )
    op_norm0 = op_norm
    op_norm1 = op_norm
    error    = 0.0

    # initialize a table to store history of Gi(x)
    Gy_avg = np.mean(Gop_memory, axis=0)  
    
    # Add the first epoch to the epoch history.
    epoch_hist.append(dict({"epoch": n_count, "error": error, "op_norm": op_norm}))
    
    # the main loop to loop over epochs.
    for k in range(total_iters):

        # update the parameters
        theta_cur = (k+1)/(k+r_shift+3)
        gamma_cur = (k+1)/(k+r_shift+1)
        eta_cur   = 2.0*beta*(k+r_shift+1)/(k+r_shift+3)

        # sample a mini-batch
        mb_id   = random.sample(full_id, mb_size)
        
        # evaluate Gy(k) and Gy(k-1) for a given mini-batch.
        y_cur   = J_op_eval(data, x_cur, rho)
        Gyi_prv = G_op_eval(data, y_prv, mb_id, mb_size)
        Gyi_cur, Gyi_mb = Gb_op_eval(data, y_cur, mb_id, mb_size)

        # compute Gz(k) for full data and Gz(k) for the mini-batch.
        Gzi_avg = np.mean([Gop_memory[i] for i in mb_id], axis=0)
        Sy_cur  = (1.0 - gamma_cur)*(Gy_avg - Gzi_avg) + Gyi_cur - gamma_cur*Gyi_prv
        Sx_cur  = Sy_cur + (x_cur - y_cur)/rho - gamma_cur*(x_prv - y_prv)/rho
        
        # store G(x) into a table.
        for i_e, i in enumerate(mb_id):
            Gop_memory[i] = Gyi_mb[i_e]
        Gy_avg   = np.mean(Gop_memory, axis=0)
        op_norm1 = la.norm(Gy_avg + (x_cur - y_cur)/rho)
        
        # update the next iterate
        x_next  = x_cur + theta_cur*(x_cur - x_prv) - eta_cur*Sx_cur

        # compute the error and operator norm.
        error   = la.norm(x_next - x_cur)
        op_norm = la.norm( Sx_cur )
        
        # only evaluate G(x) for each epoch.
        if k%n_inner_iters==0:
            n_count += 1
            #epoch_hist.append( dict({"epoch": n_count, "error": error, "op_norm": op_norm1}))
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
        x_prv  = x_cur
        y_prv  = y_cur
        x_cur  = x_next

    # end of the loop ...
    if verbose:
        print('{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=42, ), '\n')
    if k+1 >= total_iters:
        msg = "Exceed the maximum number of epochs. Increase it to run further ..."
    return dict({"opt_sol": x_cur, "message": msg, "hist": hist, "epoch_hist": epoch_hist})
