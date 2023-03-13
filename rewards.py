"""
delta (baseline, measurement, normalizer=None): calculate delta from the baseline.
It's decided that lower value is better, that's why it is (baseline - measurement) / (normalizer or baseline if not set)
"""
def delta(baseline, measurement, normalizer=None):
    if normalizer:
        return (baseline - measurement)/normalizer
    else:
        return (baseline - measurement)/baseline

"""
const_factor reward is floating-point value of metric m which should be minimized scaled by the metric n, 
deviation from which baseline upwards is undesirable, but downwards is good: (1+f(n))*(baseline(m)-measured(m))/1., 
where the f(n) is (baseline(n)-measured(n))/baseline(n), thus, for downward deviation, get the increasing of reward(m),
else -- decreasing. Increasing/decreasing are uniform as percentage of deviation.
"""
def const_factor(baseline_m, measured_m, baseline_n, measured_n):
    pass
