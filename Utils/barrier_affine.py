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


def solve_barrier_GGM(A, precision, c,
                      feasible_point,
                      con_linear,
                      con_offset,
                      step=1,
                      nstep=1000,
                      min_its=200,
                      tol=1.e-10):
    # print("Solver called")
    ## Solve the optimiaztion problem:
    ## min_u 1/2 * ( Au + c)' precision ( Au + c) + Barr(u)
    ## where Barr(u) is the barrier function for constraints of type
    ##      con_linear'u < con_offset.
    ## TODO: write sign constraints in terms of Au < b for some A, b
    conjugate_arg = precision.dot(c)
    center = A.T.dot(precision).dot(A)

    if np.asarray(c).shape == ():
        print("c:", c)
        print("|E_i| = 1")
        # center is a scalar
        # A is a (p-1)x1 vector
        # conjugate_arg is a (p-1)x1 vector
        # con_offset is a scalar
        # con_linear is a scalar
        scaling = center
        objective = lambda u: u * A.T.dot(conjugate_arg) + center*u**2 / 2. \
                              + np.log(1. + 1. / ((con_offset - con_linear*u) / scaling)).sum()
        grad = lambda u: A.T.dot(conjugate_arg) + center*u - con_linear.T.dot(
            1. / (scaling + con_offset - con_linear*u) -
            1. / (con_offset - con_linear*u))
        barrier_hessian = lambda u: con_linear * (np.diag(-1. / ((scaling + con_offset - con_linear*u) ** 2.)
                                                         + 1. / ((con_offset - con_linear*u) ** 2.))) * con_linear

    else:
        # scaling = np.sqrt(np.diag(con_linear.dot(precision).dot(con_linear.T)))
        scaling = np.sqrt(np.diag(center))
        objective = lambda u: u.T.dot(A.T).dot(conjugate_arg) + u.T.dot(center).dot(u) / 2. \
                              + np.log(1. + 1. / ((con_offset - con_linear.dot(u)) / scaling)).sum()
        grad = lambda u: A.T.dot(conjugate_arg) + center.dot(u) - con_linear.T.dot(
            1. / (scaling + con_offset - con_linear.dot(u)) -
            1. / (con_offset - con_linear.dot(u)))
        barrier_hessian = lambda u: con_linear.T.dot(np.diag(-1. / ((scaling + con_offset - con_linear.dot(u)) ** 2.)
                                                             + 1. / ((con_offset - con_linear.dot(u)) ** 2.))).dot(con_linear)

    if feasible_point is None:
        feasible_point = 1. / scaling

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

    hess = np.linalg.inv(center + barrier_hessian(current))
    if np.isnan(current_value):
        print("Laplace approximation returning nan")
    return current_value, current, hess