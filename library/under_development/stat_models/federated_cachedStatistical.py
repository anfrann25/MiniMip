import numpy as np
from library.core.statistical_model import StatisticalModel
from mini_mip_system.client.grpc_agg_client import AggregationClientInterface
from library.utils.NumpyMetricAggregator import NumpyMetricAggregator

class FederatedStatsClientCached(StatisticalModel):
    """
    Client για federated mean/metrics με cache + skip επικοινωνίας.
    """

    def __init__(
        self,
        client: AggregationClientInterface,
        warmup_rounds=3,
        tau_max=10,
        eps=1e-4,
        aggregation_interval=5,
        mix_local=0.7,
        mix_cached=0.3,
    ):
        super().__init__(client)
        self.agg = NumpyMetricAggregator(self.client)

        self.warmup_rounds = warmup_rounds
        self.tau_max = tau_max
        self.eps = eps
        self.aggregation_interval = aggregation_interval
        self.mix_local = mix_local
        self.mix_cached = mix_cached

        self.round_counter = 0
        self.cache = {"global": None}  # (value, round)
        self.prev_local = None

        self.global_history = []
        self.local_history = []
        self.did_aggregate = []

    def _update_cache(self, val: float):
        self.cache["global"] = (float(val), self.round_counter)

    def _should_aggregate(self, local_val: float) -> bool:
        if self.round_counter <= self.warmup_rounds:
            return True
        if self.cache["global"] is None:
            return True
        _, t = self.cache["global"]
        if (self.round_counter - t) > self.tau_max:
            return True
        if self.prev_local is None:
            return True
        return abs(local_val - self.prev_local) > self.eps

    def _mix_with_cached(self, local_val: float) -> float:
        if self.cache["global"] is None:
            return float(local_val)
        cached, _ = self.cache["global"]
        return float(self.mix_local * local_val + self.mix_cached * cached)

    # ---- main use cases ----

    def fit_mean(self, values: np.ndarray, num_rounds=50) -> float:
        """
        values: 1D array αριθμών. Υπολογίζει global mean.
        """
        values = np.asarray(values).reshape(-1)
        n = int(values.size)
        if n == 0:
            raise ValueError("Empty values")

        for _ in range(num_rounds):
            self.round_counter += 1

            local_sum = float(values.sum())
            local_mean = float(local_sum / n)

            self.local_history.append(local_mean)

            do_agg = self._should_aggregate(local_mean)
            if do_agg:
                global_mean = self.agg.fed_mean(local_sum, n)
                self._update_cache(global_mean)
                estimate = global_mean
            else:
                estimate = self._mix_with_cached(local_mean)

            # forced periodic sync
            if self.round_counter % self.aggregation_interval == 0:
                global_mean2 = self.agg.fed_mean(local_sum, n)
                self._update_cache(global_mean2)
                estimate = global_mean2
                do_agg = True

            self.global_history.append(float(estimate))
            self.did_aggregate.append(bool(do_agg))
            self.prev_local = local_mean

        return self.global_history[-1]

    def fit_metric_average(self, local_metric: float, weight: int, num_rounds=50) -> float:
        """
        Για metrics τύπου average (accuracy, avg loss):
          local_metric = correct/n, weight=n
        """
        local_metric = float(local_metric)
        weight = int(weight)

        for _ in range(num_rounds):
            self.round_counter += 1
            self.local_history.append(local_metric)

            do_agg = self._should_aggregate(local_metric)
            if do_agg:
                global_metric = self.agg.fed_weighted_mean_of_metric(local_metric, weight)
                self._update_cache(global_metric)
                estimate = global_metric
            else:
                estimate = self._mix_with_cached(local_metric)

            if self.round_counter % self.aggregation_interval == 0:
                global_metric2 = self.agg.fed_weighted_mean_of_metric(local_metric, weight)
                self._update_cache(global_metric2)
                estimate = global_metric2
                do_agg = True

            self.global_history.append(float(estimate))
            self.did_aggregate.append(bool(do_agg))
            self.prev_local = local_metric

        return self.global_history[-1]
    # --- required by StatisticalModel (abstract methods) ---

    def fit(self, *args, **kwargs):
        """
        Wrapper για να ικανοποιήσει το abstract interface.
        Χρησιμοποίησε:
          - fit(values=..., num_rounds=...) για mean
          - fit(local_metric=..., weight=..., num_rounds=...) για generic metric average
        """
        if "values" in kwargs:
            return self.fit_mean(kwargs["values"], num_rounds=kwargs.get("num_rounds", 50))

        if "local_metric" in kwargs and "weight" in kwargs:
            return self.fit_metric_average(
                kwargs["local_metric"],
                kwargs["weight"],
                num_rounds=kwargs.get("num_rounds", 50),
            )

        # επιτρέπει και positional: fit(values, num_rounds=...)
        if len(args) >= 1:
            values = args[0]
            return self.fit_mean(values, num_rounds=kwargs.get("num_rounds", 50))

        raise TypeError("Use fit(values=...) or fit(local_metric=..., weight=...)")

    def predict(self, *args, **kwargs):
        """
        Για statistics δεν υπάρχει κλασικό predict.
        Επιστρέφει την πιο πρόσφατη global estimate (ή cached).
        """
        if self.global_history:
            return self.global_history[-1]
        if self.cache["global"] is not None:
            return self.cache["global"][0]
        return None

