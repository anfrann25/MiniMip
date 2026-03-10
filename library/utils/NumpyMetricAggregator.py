import numpy as np
from library.utils.aggregation_client import AggregationClientInterface

class NumpyMetricAggregator:
    """
    Aggregator για scalar statistics/metrics με σωστό τρόπο,
    χρησιμοποιώντας μόνο __global_sum__.
    """

    def __init__(self, client: AggregationClientInterface):
        self.client = client

    def fed_sum_count(self, local_sum: float, local_count: float):
        """
        Στέλνει [local_sum, local_count] και παίρνει πίσω [global_sum, global_count].
        """
        payload = np.array([float(local_sum), float(local_count)], dtype=np.float64)
        ans = self.client.__global_sum__(payload.tolist())
        ans = np.asarray(ans, dtype=np.float64).reshape(-1)
        if ans.size != 2:
            raise ValueError(f"Expected 2 values back (sum,count), got {ans}")
        return float(ans[0]), float(ans[1])

    def fed_mean(self, local_sum: float, local_count: float) -> float:
        gsum, gcount = self.fed_sum_count(local_sum, local_count)
        if gcount == 0:
            return 0.0
        return float(gsum / gcount)

    def fed_weighted_mean_of_metric(self, local_metric: float, weight: float) -> float:
        """
        Για metrics που είναι average:
          global = Σ(metric_i * weight_i) / Σ(weight_i)

        Στέλνεις local_weighted_sum = metric*weight και local_weight = weight.
        """
        return self.fed_mean(local_metric * weight, weight)
