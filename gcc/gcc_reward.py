from enum import Enum
import numpy as np

class GccRew(Enum):
    RUNTIME = "Runtime",
    TextSizeBytes = "TextSizeBytes"


class RewardMetrics:
    def __init__(self, t="Runtime", value=0.0):
        self.kind = t
        self.storage = list()
        self.value = 0.0

    def reset(self):
        self.storage.clear()

    def get_last_value(self):
        return self.value


class RuntimeRewardMetrics(RewardMetrics):
    def __init__(self, t="Runtime", value=0.0, warmup_count=1, repeat_count=10, agg_func=np.mean):
        super().__init__(t, value)
        self.kind = "Runtime"
        self.warmup_count = warmup_count
        self.repeat_count = repeat_count
        self.agg_func = agg_func

    def evaluate(self, env):
        for i in range(self.warmup_count):
            env.benchmark.run()
        for i in range(self.repeat_count):
            self.storage.append(env.benchmark.run()[1])
        self.value = np.mean(self.storage)
        return self.value



class SizeRewardMetrics(RewardMetrics):
    def __init__(self, t="TextSizeBytes", value=0.0):
        super().__init__(t, value)

    def evaluate(self, env):
        self.value = env.benchmark.get_text_size()
        return self.value

    def setup(self, warmup_count=1, repeat_count=10, agg_func=np.mean):
        self.kind = "TextSizeBytes"

