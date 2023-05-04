This part of project is standalone implementation of gcc / clang + llvm opt / gnu make environments, without CompilerGym dependencies.

For the first, there is a list of shortcomings:

1) No code characterization
2) No benchmark presets
3) Compatibility with CompilerGym classes is not guaranteed.
4) No parallel execution (for now, but planned)

Nevertheless, it has a list of advantages:

1) Combined rewards from multiple metrics (size + runtime, etc)
2) optimization passes probe, without making 'step' reflected in statistics; step can be made as result of decision after a list of such probes.
3) gnu make support (some restrictions on format, but now InProgress)
4) more baseline metrics (for ex., -O1,-O2)
5)