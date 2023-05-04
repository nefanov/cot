# Compiler Optimization Tools

Compiler Optimization Tools is the set of software packages for a wide-range compiler auto-tuning tasks

## Modules

### standalone

Self-developed auto-tuning environment without any dependencies from (but inspired by) CompilerGym (see the readme in /standalone dir)

### odg_tools

Tools for manipulation with order-dependency graphs (ODG), which are frequently used in compiler phase-ordering task for subsequences of optimizing passes construction

### cgym_tools

Tools for manipulation with CompilerGym. Note: downstream ver. of CompilerGym is used (includes Ir2Vec patch, etc).

### experiment_runner

Script for running search / RL experiments from this scope:

1. #### search_algorithms
    - greedy, random search policies for size minimization under const performance
2. #### actor_critic
    - self implemented AC with historical observation & size minimization under const performance


## Supported architectures
(mainly by custom_benchmark compile and run settings)

1. AMD64 (as native)
2. AARCH64 (tested with aarch64-linux-gnu triplet)
