from scipy.stats import norm as ndist

def Z_test(y, norm_contrast, null=0, level=0.1):
    # norm_contrast'y follows N(null, 1)
    # norm_contrast'y - null ~ N(0, 1), which serves as a pivot
    # the uniform pivot we use now is 1 - CDF(pivot),
    # which should follow UNIF(0,1)
    pivot = ndist.sf(norm_contrast.dot(y) - null)
    return pivot