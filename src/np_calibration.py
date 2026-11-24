"""Normal-only NP order-statistic threshold; no positive/qualification/test labels."""
import numpy as np
from scipy.stats import binom,beta


def normal_fpr_threshold(scores,alpha=.05,delta=.05):
    values=np.asarray(scores,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():raise ValueError('Finite normal calibration scores required')
    if not 0<alpha<1 or not 0<delta<1:raise ValueError('alpha and delta must be between0 and1')
    n=len(values);bounds=binom.sf(np.arange(n),n,1-alpha);ranks=np.flatnonzero(bounds<=delta)
    if not len(ranks):raise ValueError('Insufficient normal calibration samples for the requested error bound')
    k=int(ranks[0]+1);threshold=float(np.sort(values)[k-1])
    return {'method':'normal_only_np_order_statistic','normal_calibration_images':n,'rank_one_based':k,'threshold':threshold,
            'target_normal_fpr':alpha,'violation_probability_design':delta,'binomial_tail_bound':float(bounds[k-1]),
            'normal_fpr_upper_at_design_confidence':float(beta.ppf(1-delta,n-k+1,k)),
            'calibration_observed_fp':int((values>threshold).sum()),
            'scope':'Design bound assumes an independent normal calibration set and a fixed score under matching iid acquisition; iterative validation and production distribution shift require independent acceptance, not a universal guarantee'}
