"""
delta (baseline, measurement, normalizer=None): calculate delta from the baseline.
It's decided that lower value is better, that's why it is (baseline - measurement) / (normalizer or baseline if not set)
"""
import math


def delta(baseline, measurement, normalizer=None):
    if normalizer:
        return (baseline - measurement) / normalizer
    else:
        return (baseline - measurement) / baseline


"""
const_factor reward is floating-point value of metric m which should be minimized scaled by the metric n, 
deviation from which baseline upwards is undesirable, but downwards is good: 
Let f(n) is (baseline(n)-measured(n))/baseline(n), d(m) = (baseline(m)-measured(m))/1.

(1+f(n))*d(m), if d(m) > 0 and ( f(n) > 0 or (f(n) <= 0 and |f(n)| < threshold))
  (1-f(n))*d(m), if d(m) <= 0
  else 0.
,
 
 thus, for downward deviation, get the increasing of reward(m),
else -- decreasing. Increasing/decreasing are uniform as percentage of deviation, but thresholded in goal of n fixing.
if threshold is set and upward deviation is bigger than threshold, and m is worst than its baseline, 
"""


def const_factor_threshold(baseline_m, measured_m, baseline_n, measured_n, threshold=0.01, m_norm = None):
    f_n = (baseline_n - measured_n) / baseline_n
    d = (baseline_m - measured_m)
    if not m_norm:
        d=d/baseline_m
    else:
        d=d/m_norm
    if d > 0:
        if (f_n > 0) or (f_n <= 0 and math.fabs(f_n) < threshold):
            return (1.+f_n) * d
        else:
            return 0.
    else:
        return (1-f_n) * d


if __name__ == '__main__':
    print(const_factor_threshold(100, 90, 10, 100)) # should return 0.0 -- ten times n degradation
    print(const_factor_threshold(100, 102, 100, 100))  # should return -2
    print(const_factor_threshold(100, 98, 100, 100))  # should return 2
    print(const_factor_threshold(100, 110, 10, 100)) # should return very negative reward
