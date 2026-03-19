import numpy as np
from library.core.statistical_model import StatisticalModel
from mini_mip_system.client.grpc_agg_client import AggregationClientInterface
from library.utils.NumpyMetricAggregator import NumpyMetricAggregator


class FederatedStatsClientCached(StatisticalModel):
    """
    Client για federated mean/metrics με cache + skip επικοινωνίας.
    Cache ανά metric.
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

        if not np.isclose(self.mix_local + self.mix_cached, 1.0):
            raise ValueError("mix_local + mix_cached must be 1.0")

        self.round_counter = 0

        #cache ana metric
        self.cache = {
            "mean": None,
            "metric_average": None,
            "variance": None,
            "std": None,
            "rms": None,
            "covariance": None,
            "pearson": None,
            "regression": None,
        }

        self.prev_local = {
            "mean": None,
            "metric_average": None,
            "variance": None,
            "std": None,
            "rms": None,
            "covariance": None,
            "pearson": None,
            "regression": None,
        }

        self.global_history = []
        self.local_history = []
        self.did_aggregate = []

    # -------------------- generic cache helpers --------------------

    def _update_cache(self, metric_name: str, val):
        self.cache[metric_name] = (val, self.round_counter)

    def _get_cached_value(self, metric_name: str):
        if self.cache.get(metric_name) is None:
            return None
        return self.cache[metric_name][0]

    def _should_aggregate(self, metric_name: str, local_val: float) -> bool:
        if self.round_counter <= self.warmup_rounds:
            return True

        if self.cache.get(metric_name) is None:
            return True

        _, t = self.cache[metric_name]
        if (self.round_counter - t) > self.tau_max:
            return True

        prev_val = self.prev_local.get(metric_name)
        if prev_val is None:
            return True

        return abs(local_val - prev_val) > self.eps

    def _mix_with_cached(self, metric_name: str, local_val: float) -> float:
        cached_entry = self.cache.get(metric_name)
        if cached_entry is None:
            return float(local_val)

        cached_val, _ = cached_entry
        return float(self.mix_local * local_val + self.mix_cached * cached_val)

    def _record_step(self, local_val, estimate, did_agg: bool, metric_name: str):
        self.local_history.append((metric_name, local_val))
        self.global_history.append((metric_name, estimate))
        self.did_aggregate.append((metric_name, bool(did_agg)))
        self.prev_local[metric_name] = local_val

    def _validate_1d(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values).reshape(-1)
        if values.size == 0:
            raise ValueError("Empty values")
        return values

    def _validate_pair(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        if x.size == 0 or y.size == 0:
            raise ValueError("Empty values")
        if x.shape != y.shape:
            raise ValueError("x and y must have same shape")
        return x, y

    def _global_mean_from_values(self, values: np.ndarray) -> float:
        values = np.asarray(values).reshape(-1)
        n = int(values.size)
        if n == 0:
            raise ValueError("Empty values")
        local_sum = float(values.sum())
        return float(self.agg.fed_mean(local_sum, n))

    def _fit_scalar_metric(self, metric_name: str, local_val: float, agg_fn, num_rounds=50) -> float:
        local_val = float(local_val)

        for _ in range(num_rounds):
            self.round_counter += 1

            do_agg = self._should_aggregate(metric_name, local_val)

            if do_agg:
                global_val = float(agg_fn())
                self._update_cache(metric_name, global_val)
                estimate = global_val
            else:
                estimate = self._mix_with_cached(metric_name, local_val)

            if self.round_counter % self.aggregation_interval == 0:
                global_val2 = float(agg_fn())
                self._update_cache(metric_name, global_val2)
                estimate = global_val2
                do_agg = True

            self._record_step(local_val, estimate, do_agg, metric_name)

        return self._get_cached_value(metric_name)

    # -------------------- main use cases --------------------

    def fit_variance(self, values: np.ndarray, num_rounds=50) -> float:
        values = self._validate_1d(values)
        local_var = float(np.var(values))

        def aggregate_variance():
            ex = self._global_mean_from_values(values)
            ex2 = self._global_mean_from_values(values ** 2)
            return float(max(ex2 - ex * ex, 0.0))

        return self._fit_scalar_metric("variance", local_var, aggregate_variance, num_rounds=num_rounds)

    def fit_std(self, values: np.ndarray, num_rounds=50) -> float:
        values = self._validate_1d(values)
        local_std = float(np.std(values))

        def aggregate_std():
            ex = self._global_mean_from_values(values)
            ex2 = self._global_mean_from_values(values ** 2)
            var = max(ex2 - ex * ex, 0.0)
            return float(np.sqrt(var))

        return self._fit_scalar_metric("std", local_std, aggregate_std, num_rounds=num_rounds)

    def fit_rms(self, values: np.ndarray, num_rounds=50) -> float:
        values = self._validate_1d(values)
        local_rms = float(np.sqrt(np.mean(values ** 2)))

        def aggregate_rms():
            ex2 = self._global_mean_from_values(values ** 2)
            return float(np.sqrt(ex2))

        return self._fit_scalar_metric("rms", local_rms, aggregate_rms, num_rounds=num_rounds)

    def fit_covariance(self, x: np.ndarray, y: np.ndarray, num_rounds=50) -> float:
        x, y = self._validate_pair(x, y)
        local_cov = float(np.mean((x - x.mean()) * (y - y.mean())))

        def aggregate_cov():
            ex = self._global_mean_from_values(x)
            ey = self._global_mean_from_values(y)
            exy = self._global_mean_from_values(x * y)
            return float(exy - ex * ey)

        return self._fit_scalar_metric("covariance", local_cov, aggregate_cov, num_rounds=num_rounds)

    def fit_pearson(self, x: np.ndarray, y: np.ndarray, num_rounds=50) -> float:
        x, y = self._validate_pair(x, y)

        local_std_x = float(np.std(x))
        local_std_y = float(np.std(y))

        if local_std_x == 0 or local_std_y == 0:
            local_pearson = 0.0
        else:
            local_cov = float(np.mean((x - x.mean()) * (y - y.mean())))
            local_pearson = float(local_cov / (local_std_x * local_std_y))

        def aggregate_pearson():
            ex = self._global_mean_from_values(x)
            ey = self._global_mean_from_values(y)
            ex2 = self._global_mean_from_values(x ** 2)
            ey2 = self._global_mean_from_values(y ** 2)
            exy = self._global_mean_from_values(x * y)

            var_x = max(ex2 - ex * ex, 0.0)
            var_y = max(ey2 - ey * ey, 0.0)
            if var_x == 0 or var_y == 0:
                return 0.0

            cov = exy - ex * ey
            return float(cov / np.sqrt(var_x * var_y))

        return self._fit_scalar_metric("pearson", local_pearson, aggregate_pearson, num_rounds=num_rounds)

    def fit_regression(self, x: np.ndarray, y: np.ndarray, num_rounds=50):
        x, y = self._validate_pair(x, y)

        local_var_x = float(np.var(x))
        if local_var_x == 0:
            local_val = (0.0, float(np.mean(y)))
        else:
            local_cov = float(np.mean((x - x.mean()) * (y - y.mean())))
            slope = float(local_cov / local_var_x)
            intercept = float(np.mean(y) - slope * np.mean(x))
            local_val = (slope, intercept)

        for _ in range(num_rounds):
            self.round_counter += 1

            do_agg = True
            if self.round_counter > self.warmup_rounds:
                if self.cache.get("regression") is None:
                    do_agg = True
                else:
                    _, t = self.cache["regression"]
                    if (self.round_counter - t) > self.tau_max:
                        do_agg = True
                    else:
                        prev_val = self.prev_local.get("regression")
                        if prev_val is None:
                            do_agg = True
                        else:
                            diff = max(
                                abs(local_val[0] - prev_val[0]),
                                abs(local_val[1] - prev_val[1]),
                            )
                            do_agg = diff > self.eps

            def aggregate_reg():
                ex = self._global_mean_from_values(x)
                ey = self._global_mean_from_values(y)
                ex2 = self._global_mean_from_values(x ** 2)
                exy = self._global_mean_from_values(x * y)

                var_x = max(ex2 - ex * ex, 0.0)
                if var_x == 0:
                    return (0.0, float(ey))

                cov_xy = exy - ex * ey
                slope = float(cov_xy / var_x)
                intercept = float(ey - slope * ex)
                return (slope, intercept)

            if do_agg:
                reg = aggregate_reg()
                self._update_cache("regression", reg)
                estimate = reg
            else:
                cached = self._get_cached_value("regression")
                if cached is None:
                    estimate = local_val
                else:
                    estimate = (
                        self.mix_local * local_val[0] + self.mix_cached * cached[0],
                        self.mix_local * local_val[1] + self.mix_cached * cached[1],
                    )

            if self.round_counter % self.aggregation_interval == 0:
                reg2 = aggregate_reg()
                self._update_cache("regression", reg2)
                estimate = reg2
                do_agg = True

            self._record_step(local_val, estimate, do_agg, "regression")

        return self._get_cached_value("regression")

    def fit_mean(self, values: np.ndarray, num_rounds=50) -> float:
        """
        values: 1D array αριθμών. Υπολογίζει global mean.
        """
        values = np.asarray(values).reshape(-1)
        n = int(values.size)

        if n == 0:
            raise ValueError("Empty values")

        local_sum = float(values.sum())
        local_mean = float(local_sum / n)

        for _ in range(num_rounds):
            self.round_counter += 1

            do_agg = self._should_aggregate("mean", local_mean)

            if do_agg:
                global_mean = self.agg.fed_mean(local_sum, n)
                self._update_cache("mean", float(global_mean))
                estimate = float(global_mean)
            else:
                estimate = self._mix_with_cached("mean", local_mean)

            # forced periodic sync
            if self.round_counter % self.aggregation_interval == 0:
                global_mean2 = self.agg.fed_mean(local_sum, n)
                self._update_cache("mean", float(global_mean2))
                estimate = float(global_mean2)
                do_agg = True

            self._record_step(local_mean, estimate, do_agg, "mean")

        return self._get_cached_value("mean")

    def fit_metric_average(self, local_metric: float, weight: int, num_rounds=50) -> float:
        """
        Για metrics τύπου average (accuracy, avg loss):
          local_metric = correct/n, weight=n
        """
        local_metric = float(local_metric)
        weight = int(weight)

        if weight <= 0:
            raise ValueError("weight must be > 0")

        for _ in range(num_rounds):
            self.round_counter += 1

            do_agg = self._should_aggregate("metric_average", local_metric)

            if do_agg:
                global_metric = self.agg.fed_weighted_mean_of_metric(local_metric, weight)
                self._update_cache("metric_average", float(global_metric))
                estimate = float(global_metric)
            else:
                estimate = self._mix_with_cached("metric_average", local_metric)

            if self.round_counter % self.aggregation_interval == 0:
                global_metric2 = self.agg.fed_weighted_mean_of_metric(local_metric, weight)
                self._update_cache("metric_average", float(global_metric2))
                estimate = float(global_metric2)
                do_agg = True

            self._record_step(local_metric, estimate, do_agg, "metric_average")

        return self._get_cached_value("metric_average")

    # -------------------- abstract interface --------------------

    def fit(self, *args, **kwargs):
        if "values" in kwargs and kwargs.get("metric", "mean") == "mean":
            return self.fit_mean(kwargs["values"], num_rounds=kwargs.get("num_rounds", 50))

        if "values" in kwargs and kwargs.get("metric") == "variance":
            return self.fit_variance(kwargs["values"], num_rounds=kwargs.get("num_rounds", 50))

        if "values" in kwargs and kwargs.get("metric") == "std":
            return self.fit_std(kwargs["values"], num_rounds=kwargs.get("num_rounds", 50))

        if "values" in kwargs and kwargs.get("metric") == "rms":
            return self.fit_rms(kwargs["values"], num_rounds=kwargs.get("num_rounds", 50))

        if "x" in kwargs and "y" in kwargs and kwargs.get("metric") == "covariance":
            return self.fit_covariance(kwargs["x"], kwargs["y"], num_rounds=kwargs.get("num_rounds", 50))

        if "x" in kwargs and "y" in kwargs and kwargs.get("metric") == "pearson":
            return self.fit_pearson(kwargs["x"], kwargs["y"], num_rounds=kwargs.get("num_rounds", 50))

        if "x" in kwargs and "y" in kwargs and kwargs.get("metric") == "regression":
            return self.fit_regression(kwargs["x"], kwargs["y"], num_rounds=kwargs.get("num_rounds", 50))

        if "local_metric" in kwargs and "weight" in kwargs:
            return self.fit_metric_average(
                kwargs["local_metric"],
                kwargs["weight"],
                num_rounds=kwargs.get("num_rounds", 50),
            )

        if len(args) >= 1:
            values = args[0]
            return self.fit_mean(values, num_rounds=kwargs.get("num_rounds", 50))

        raise TypeError("Unsupported fit call")

    def predict(self, metric_name="mean", *args, **kwargs):
        """
        Επιστρέφει την πιο πρόσφατη cached τιμή για το metric.
        """
        cached = self.cache.get(metric_name)
        if cached is not None:
            return cached[0]
        return None