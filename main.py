import cgym_tools as cgt
import odg_tools as odg

if __name__ == '__main__':
    # test #1
    odg4 = odg.ODG(name="O3_part").fromFullSeq(cgt.Experiment(
        "llvm-v0",
        bench="cbench-v1/qsort",
        observation_space="Ir2vecFlowAware",
        reward_space="IrInstructionCountO3",
    ).getActions()[:10])
    odg4.visualize(True)
