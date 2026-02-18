import numpy as np
from library.core.statistical_model import StatisticalModel
from library.utils.numpy_aggregator import NumpyAggregator
from mini_mip_system.client.grpc_agg_client import AggregationClientInterface

def _abs(x):
    return float(abs(x))

class FederatedMetricClientCached(StatisticalModel):
    """
    Federated average για metrics που είναι average πάνω σε δείγματα.
    Στέλνεις (local_metric, weight=n) και παίρνεις global weighted average.
    """
    def __init__(
        self,
        client: AggregationClientInterface,
        aggregation_interval=3,
        warmup_rounds=5,
        tau_max=10,
        eps=1e-3,
        mix_local=0.7,
        mix_cached=0.3,
    ):
        super().__init__(client)
        self.agg = NumpyAggregator(self.client)

        self.aggregation_interval = aggregation_interval
        self.warmup_rounds = warmup_rounds
        self.tau_max = tau_max
        self.eps = eps
        self.mix_local = mix_local
        self.mix_cached = mix_cached

        self.cache = {"global_metric": None}  # (value, round)
        self.round_counter = 0
        self.prev_local_metric = None

        self.global_metric_history = []
        self.local_metric_history = []
        self.did_aggregate_history = []

    def _update_cache(self, global_metric: float):
        self.cache["global_metric"] = (float(global_metric), self.round_counter)

    def _should_aggregate(self, local_metric: float):
        if self.round_counter <= self.warmup_rounds:
            return True

        if self.cache["global_metric"] is None:
            return True

        _, t_cached = self.cache["global_metric"]
        if (self.round_counter - t_cached) > self.tau_max:
            return True

        if self.prev_local_metric is None:
            return True

        d = _abs(local_metric - self.prev_local_metric)
        return d > self.eps

    def _mix_with_cached(self, local_metric: float):
        if self.cache["global_metric"] is None:
            return float(local_metric)
        cached, _ = self.cache["global_metric"]
        return float(self.mix_local * local_metric + self.mix_cached * cached)

    def fit_mean(self, values: np.ndarray, num_rounds=100):
        """
        values: 1D array αριθμών (π.χ. feature values).
        Υπολογίζει global mean μέσω federated weighted avg των local means.
        """
        values = np.asarray(values).reshape(-1)
        n = int(values.size)
        if n == 0:
            raise ValueError("Empty values array")

        for _ in range(num_rounds):
            self.round_counter += 1

            local_mean = float(values.mean())      # local metric
            weight = n                             # weight

            self.local_metric_history.append(local_mean)

            do_agg = self._should_aggregate(local_mean)

            if do_agg:
                # fed_weighted_avg(local_metric, weight)
                # στέλνουμε array για να είμαστε συμβατοί με numpy aggregator
                global_mean = self.agg.fed_weighted_avg(np.array([local_mean]), weight)
                global_mean = float(np.asarray(global_mean).reshape(-1)[0])

                self._update_cache(global_mean)
                estimate = global_mean
            else:
                estimate = self._mix_with_cached(local_mean)

            # Forced periodic sync (αν το θες)
            if self.round_counter % self.aggregation_interval == 0:
                global_mean2 = self.agg.fed_weighted_avg(np.array([local_mean]), weight)
                global_mean2 = float(np.asarray(global_mean2).reshape(-1)[0])
                self._update_cache(global_mean2)
                estimate = global_mean2
                do_agg = True

            self.global_metric_history.append(float(estimate))
            self.did_aggregate_history.append(bool(do_agg))
            self.prev_local_metric = local_mean

        return self.global_metric_history[-1]
