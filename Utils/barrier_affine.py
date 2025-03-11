import numpy as np

def solve_barrier_affine_py(conjugate_arg,
                            precision,
                            feasible_point,
                            con_linear,
                            con_offset,
                            step=1,
                            nstep=1000,
                            min_its=200,
                            tol=1.e-10):

    scaling = np.sqrt(np.diag(con_linear.dot(precision).dot(con_linear.T)))

    if feasible_point is None:
        feasible_point = 1. / scaling

    objective = lambda u: -u.T.dot(conjugate_arg) + u.T.dot(precision).dot(u)/2. \
                          + np.log(1.+ 1./((con_offset - con_linear.dot(u))/ scaling)).sum()
    grad = lambda u: -conjugate_arg + precision.dot(u) - con_linear.T.dot(1./(scaling + con_offset - con_linear.dot(u)) -
                                                                       1./(con_offset - con_linear.dot(u)))
    barrier_hessian = lambda u: con_linear.T.dot(np.diag(-1./((scaling + con_offset-con_linear.dot(u))**2.)
                                                 + 1./((con_offset-con_linear.dot(u))**2.))).dot(con_linear)

    current = feasible_point
    current_value = -1000 # change from np.inf -> -1000

    for itercount in range(nstep):
        cur_grad = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            if np.all(con_offset-con_linear.dot(proposal) > 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')

        # make sure proposal is a descent

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5
            if count >= 20:
                if not (np.isnan(proposed_value) or np.isnan(current_value)):
                    break
                else:
                    raise ValueError('value is NaN: %f, %f' % (proposed_value, current_value))

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value) and itercount >= min_its:
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

        if itercount % 100 == 0:
            print(itercount)


    hess = np.linalg.inv(precision + barrier_hessian(current))
    return current_value, current, hess

def solve_barrier_nonneg(conjugate_arg,
                         precision,
                         feasible_point=None,
                         step=1,
                         nstep=1000,
                         tol=1.e-8):

    scaling = np.sqrt(np.diag(precision))

    if feasible_point is None:
        feasible_point = 1. / scaling

    objective = lambda u: -u.T.dot(conjugate_arg) + u.T.dot(precision).dot(u) / 2. + np.log(
        1. + 1. / (u / scaling)).sum()
    grad = lambda u: -conjugate_arg + precision.dot(u) + (1. / (scaling + u) - 1. / u)
    barrier_hessian = lambda u: (-1. / ((scaling + u) ** 2.) + 1. / (u ** 2.))

    current = feasible_point
    current_value = np.inf

    for itercount in range(nstep):
        cur_grad = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            if np.all(proposal > 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')

        # make sure proposal is a descent

        count = 0
        while True:
            proposal = current - step * cur_grad
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5
            if count >= 20:
                if not (np.isnan(proposed_value) or np.isnan(current_value)):
                    break
                else:
                    raise ValueError('value is NaN: %f, %f' % (proposed_value, current_value))

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    hess = np.linalg.inv(precision + np.diag(barrier_hessian(current)))
    return current_value, current, hess


def solve_barrier_tree(Q, precision,
                      feasible_point,
                       con_linear,
                       con_offset,
                      step=1,
                      nstep=1000,
                      min_its=200,
                      tol=1.e-10):
    ## Solve the optimiaztion problem:
    ## min_u 1/2 * ( u - Q)' precision ( u - Q) + Barr(u)
    ## where Barr(u) is the barrier function for constraints of type
    ##      con_linear'u < con_offset;
    ## and in particular con_linear = [I -I]', con_offset = [0' -offset']'.
    conjugate_arg = -1 * precision.dot(Q)
    center = precision

    if np.asarray(Q).shape == ():
        # center is a scalar
        # A is a (p-1)x1 vector
        # conjugate_arg is a (p-1)x1 vector
        # con_offset is a scalar
        # con_linear is a scalar
        scaling = center
        objective = lambda u: u * conjugate_arg + center * u ** 2 / 2. \
                              + np.log(1. + 1. / ((con_offset - con_linear*u) / scaling)).sum()
        grad = lambda u: (conjugate_arg + center*u
                          - con_linear.T.dot(
                            1. / (scaling + con_offset - con_linear*u) -
                            1. / (con_offset - con_linear*u)))
        barrier_hessian = lambda u: con_linear * (np.diag(-1. / ((scaling + con_offset - con_linear*u) ** 2.)
                                                         + 1. / ((con_offset - con_linear*u) ** 2.))) * con_linear

    else:
        #scaling = np.sqrt(np.diag(con_linear.dot(precision).dot(con_linear.T)))
        scaling = np.sqrt(np.diag(center))
        scaling_dim = scaling.shape[0]
        scaling_2d = np.zeros(scaling_dim * 2)
        scaling_2d[0:scaling_dim] = scaling
        scaling_2d[scaling_dim:] = scaling
        objective = lambda u: u.T.dot(conjugate_arg) + u.T.dot(center).dot(u) / 2. \
                              + np.log(1. + 1. / ((con_offset - con_linear.dot(u)) / scaling_2d)).sum()
        grad = lambda u: (conjugate_arg) + center.dot(u)  - con_linear.T.dot(
            1. / (scaling_2d + con_offset - con_linear.dot(u)) -
            1. / (con_offset - con_linear.dot(u)))
        barrier_hessian = lambda u: con_linear.T.dot(np.diag(-1. / ((scaling_2d + con_offset - con_linear.dot(u)) ** 2.)
                                                             + 1. / ((con_offset - con_linear.dot(u)) ** 2.))).dot(con_linear)

    if feasible_point is None:
        feasible_point = - 1. / scaling

    current = feasible_point
    current_value = -1000 # change from np.inf -> -1000

    for itercount in range(nstep):
        cur_grad = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            if np.all(proposal < 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')

        # make sure proposal is a descent

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5
            if count >= 20:
                if not (np.isnan(proposed_value) or np.isnan(current_value)):
                    break
                else:
                    raise ValueError('value is NaN: %f, %f' % (proposed_value, current_value))

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value) and itercount >= min_its:
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    hess = np.linalg.inv(center + barrier_hessian(current))
    if np.isnan(current_value):
        print("Laplace approximation returning nan")

    ## Assess the impact of the barrier function
    obj = proposal.T.dot(conjugate_arg) + proposal.T.dot(center).dot(proposal) / 2.
    barr = np.log(1. + 1. / ((con_offset - con_linear.dot(proposal)) / scaling_2d)).sum()

    print("proposal norm", np.linalg.norm(proposal))
    print("proposal max/min:", np.max(proposal), np.min(proposal))
    print("constraint max/min:", np.max(con_offset), np.min(con_offset))
    print("cross", proposal.T.dot(conjugate_arg))
    print("quad", proposal.T.dot(center).dot(proposal) / 2.)
    print("barr", (barr))
    print("obj", (barr + obj))
    return current_value, current, hess

def solve_barrier_tree_nonneg_lb(Q, precision,
                                 feasible_point,
                                 lb,
                                 step=1,
                                 nstep=1000,
                                 min_its=200,
                                 tol=1.e-10):
    ## Solve the optimiaztion problem:
    ## min_u 1/2 * ( u - Q)' precision ( u - Q) + Barr(u)
    ## where Barr(u) is the barrier function for constraints of type
    ##      con_linear'u < con_offset;
    ## and in particular con_linear = [I -I]', con_offset = [0' -offset']'.
    conjugate_arg = -1 * precision.dot(Q)
    center = precision
    # TODO: Debug

    if np.asarray(Q).shape == ():
        # center is a scalar
        # A is a (p-1)x1 vector
        # conjugate_arg is a (p-1)x1 vector
        # con_offset is a scalar
        # con_linear is a scalar
        scaling = center
        objective = lambda u: u * conjugate_arg + center * u ** 2 / 2. \
                              + np.log(1. + 1. / ( - u / scaling)).sum() \
                              + np.log(1. - 1. / ((lb - u) / scaling)).sum()
        grad = lambda u: (conjugate_arg + center * u
                          - (1. / (scaling - u) + 1. / (u))
                          - scaling/((lb-u)*(lb-scaling-u)) )

    else:
        #scaling = np.sqrt(np.diag(con_linear.dot(precision).dot(con_linear.T)))
        scaling = np.sqrt(np.diag(center))
        objective = lambda u: u.T.dot(conjugate_arg) + u.T.dot(center).dot(u) / 2. \
                              + np.log(1. + 1. / (( - u) / scaling)).sum() \
                              + np.log(1. - 1. / ((lb - u) / scaling)).sum()
        grad = lambda u: ((conjugate_arg) + center.dot(u) -
                          ( 1. / (scaling  - u) + 1. / u) -
                           scaling/((lb-u)*(lb-scaling-u)) )

    if feasible_point is None:
        feasible_point = - 1. / scaling
        if np.min(feasible_point) <= lb:
            feasible_point = np.zeros(feasible_point) + 1/2 * lb

    current = feasible_point
    current_value = -1000 # change from np.inf -> -1000

    for itercount in range(nstep):
        cur_grad = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            if np.all(proposal < 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')

        # make sure proposal is a descent

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5
            if count >= 20:
                if not (np.isnan(proposed_value) or np.isnan(current_value)):
                    break
                else:
                    raise ValueError('value is NaN: %f, %f' % (proposed_value, current_value))

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value) and itercount >= min_its:
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    #hess = np.linalg.inv(center + barrier_hessian(current))
    if np.isnan(current_value):
        print("Laplace approximation returning nan")

    assert np.min(proposal) > lb
    assert np.max(proposal) <= 0

    ## Assess the impact of the barrier function
    #obj = proposal.T.dot(conjugate_arg) + proposal.T.dot(center).dot(proposal) / 2.
    #barr = np.log(1. + 1. / ( - proposal / scaling)).sum()

    #print("barr", (barr))
    #print("obj", (barr + obj))
    return current_value, current, None#hess

def solve_barrier_tree_box_PGD(Q, precision,
                                 feasible_point,
                                 lb,
                                 step=1,
                                 nstep=1000,
                                 min_its=200,
                                 tol=1.e-10):
    ## Solve the optimiaztion problem:
    ## min_u 1/2 * ( u - Q)' precision ( u - Q) + Barr(u)
    ## where Barr(u) is the barrier function for constraints of type
    ##      con_linear'u < con_offset;
    ## and in particular con_linear = [I -I]', con_offset = [0' -offset']'.
    conjugate_arg = -1 * precision.dot(Q)
    center = precision
    proj = lambda u: np.maximum(np.minimum(u,0), lb)

    if np.asarray(Q).shape == ():
        # center is a scalar
        # A is a (p-1)x1 vector
        # conjugate_arg is a (p-1)x1 vector
        # con_offset is a scalar
        # con_linear is a scalar
        scaling = center
        objective = lambda u: u * conjugate_arg + center * u ** 2 / 2.
        grad = lambda u: (conjugate_arg + center * u)

    else:
        #scaling = np.sqrt(np.diag(con_linear.dot(precision).dot(con_linear.T)))
        scaling = np.sqrt(np.diag(center))
        objective = lambda u: u.T.dot(conjugate_arg) + u.T.dot(center).dot(u) / 2.
        grad = lambda u: ((conjugate_arg) + center.dot(u))

    if feasible_point is None:
        feasible_point = - 1. / scaling
        if np.min(feasible_point) <= lb:
            feasible_point = np.zeros_like(feasible_point) + 1/2 * lb

    current = feasible_point
    current_value = -1000 # change from np.inf -> -1000

    for itercount in range(nstep):
        cur_grad = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = proj(current - step * cur_grad)
            if np.all(proposal < 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')

        # make sure proposal is a descent

        count = 0
        while True:
            count += 1
            proposal = proj(current - step * cur_grad)
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5
            if count >= 20:
                if not (np.isnan(proposed_value) or np.isnan(current_value)):
                    break
                else:
                    raise ValueError('value is NaN: %f, %f' % (proposed_value, current_value))

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value) and itercount >= min_its:
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    #hess = np.linalg.inv(center + barrier_hessian(current))
    if np.isnan(current_value):
        print("Laplace approximation returning nan")

    assert np.min(proposal) >= lb
    assert np.max(proposal) <= 0

    ## Assess the impact of the barrier function
    #obj = proposal.T.dot(conjugate_arg) + proposal.T.dot(center).dot(proposal) / 2.
    #barr = np.log(1. + 1. / ( - proposal / scaling)).sum()

    #print("barr", (barr))
    #print("obj", (barr + obj))
    return current_value, current, None#hess

# Helper functions for PGD
def A_mv(z):
    """
    Compute A*z for A = I - (1*1^T)/(k+1), in O(k) time.
    """
    # z in R^k
    k = z.shape[0]
    # sum_z = 1^T z
    sum_z = np.sum(z)
    # A*z = z - (sum_z)/(k+1) * 1
    return z - (sum_z / (k + 1)) * np.ones(k)

def f_val(x, u):
    """
    Objective f(x) = (x - u)^T A (x - u), using the A_mv operator.
    """
    z = x - u
    Az = A_mv(z)
    return np.dot(z, Az)  # z^T (A z)

def grad_f(x, u):
    """
    Gradient of f(x) = 2 * A (x - u).
    """
    z = x - u
    return 2.0 * A_mv(z)

def project_orthant(x):
    """
    Projection onto {x : x <= 0}, i.e. elementwise min(x, 0).
    """
    return np.minimum(x, 0)

def projected_gradient_descent(u, x_init, step_size=0.49, max_iter=1000, tol=1e-6, verbose=False):
    """
    PGD to solve:
       minimize (x - u)^T A (x - u)
       subject to x <= 0
    where A = I - (1*1^T)/(k+1).

    Args:
      u (ndarray): The vector u in R^k.
      x_init (ndarray): Starting guess for x in R^k.
      step_size (float): Step size (must be < 1/2 for guaranteed convergence).
      max_iter (int): Maximum PGD iterations.
      tol (float): Convergence threshold for ||x^{new} - x^{old}||.
      verbose (bool): If True, prints iteration info.

    Returns:
      x (ndarray): Approximate minimizer x^*.
      objs (list): The objective value at each iteration.
    """
    x = x_init.copy()
    objs = []
    for t in range(max_iter):
        # Compute gradient
        g = grad_f(x, u)

        # Gradient step
        x_new = x - step_size * g

        # Projection onto x <= 0
        x_new = project_orthant(x_new)

        # Check change
        dist = np.linalg.norm(x_new - x, 2)
        x = x_new

        # Compute objective
        obj_val = f_val(x, u)
        objs.append(obj_val)

        if verbose and (t % 50 == 0 or t == max_iter - 1):
            print(f"iter={t}, obj={obj_val:.6g}, step={dist:.3g}")

        # Stopping criterion
        if dist < tol:
            break

    return x, objs
def solve_barrier_tree_nonneg_PGD(implied_mean, noise_sd=None, init_point=None):
    ## Solve the optimiaztion problem:
    ## min_u 1/2 * ( u - Q)' precision ( u - Q)
    ## where Barr(u) is the barrier function for constraints of type
    ##      con_linear'u < con_offset;
    ## and in particular con_linear = [I -I]', con_offset = [0' -offset']'.
    if init_point is None:
        init_point = project_orthant(implied_mean)

    current, traj = projected_gradient_descent(u=implied_mean,
                                               x_init=init_point)

    assert np.max(current) <= 0
    return -traj[-1]/(2*noise_sd**2), current, None#hess

def solve_barrier_tree_nonneg(Q, precision,
                              feasible_point,
                              step=1,
                              nstep=1000,
                              min_its=200,
                              tol=1.e-10):
    ## Solve the optimiaztion problem:
    ## min_u 1/2 * ( u - Q)' precision ( u - Q) + Barr(u)
    ## where Barr(u) is the barrier function for constraints of type
    ##      con_linear'u < con_offset;
    ## and in particular con_linear = [I -I]', con_offset = [0' -offset']'.
    conjugate_arg = -1 * precision.dot(Q)
    center = precision
    # TODO: Debug

    if np.asarray(Q).shape == ():
        # center is a scalar
        # A is a (p-1)x1 vector
        # conjugate_arg is a (p-1)x1 vector
        # con_offset is a scalar
        # con_linear is a scalar
        scaling = center
        objective = lambda u: u * conjugate_arg + center * u ** 2 / 2. \
                              + np.log(1. + 1. / ( - u / scaling)).sum()
        grad = lambda u: (conjugate_arg + center * u
                          - (1. / (scaling - u) -
                             1. / (- u)))
        barrier_hessian = lambda u: (np.diag(-1. / ((scaling - u) ** 2.)
                                             + 1. / (( - u) ** 2.)))

    else:
        #scaling = np.sqrt(np.diag(con_linear.dot(precision).dot(con_linear.T)))
        scaling = np.sqrt(np.diag(center))
        objective = lambda u: u.T.dot(conjugate_arg) + u.T.dot(center).dot(u) / 2. \
                              + np.log(1. + 1. / (( - u) / scaling)).sum()
        grad = lambda u: (conjugate_arg) + center.dot(u) - (
            1. / (scaling  - u) -
            1. / ( -u))
        barrier_hessian = lambda u: (np.diag(-1. / ((scaling - u) ** 2.)
                                                             + 1. / (( - u) ** 2.)))
    if feasible_point is None:
        feasible_point = - 1. / scaling

    current = feasible_point
    current_value = -1000 # change from np.inf -> -1000

    for itercount in range(nstep):
        cur_grad = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            if np.all(proposal < 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')

        # make sure proposal is a descent

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5
            if count >= 20:
                if not (np.isnan(proposed_value) or np.isnan(current_value)):
                    break
                else:
                    raise ValueError('value is NaN: %f, %f' % (proposed_value, current_value))

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value) and itercount >= min_its:
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    hess = np.linalg.inv(center + barrier_hessian(current))
    if np.isnan(current_value):
        print("Laplace approximation returning nan")

    ## Assess the impact of the barrier function
    obj = proposal.T.dot(conjugate_arg) + proposal.T.dot(center).dot(proposal) / 2.
    barr = np.log(1. + 1. / ( - proposal / scaling)).sum()

    print("barr", (barr))
    print("obj", (barr + obj))
    return current_value, current, hess