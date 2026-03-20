import numpy as np
from library.core.statistical_model import StatisticalModel
from mini_mip_system.client.grpc_agg_client import AggregationClientInterface
from library.utils.NumpyMetricAggregator import NumpyMetricAggregator


class FederatedStatsClient(StatisticalModel):
    """
    Client για federated mean/metrics χωρίς cache.
    Σε κάθε round γίνεται κανονικό aggregation.
    """

    def __init__(self, client: AggregationClientInterface):
        super().__init__(client)
        self.agg = NumpyMetricAggregator(self.client)

        self.round_counter = 0
        self.last_results = {
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

    # -------------------- helpers --------------------

    def _record_step(self, local_val, global_val, metric_name: str):
        self.local_history.append((metric_name, local_val))
        self.global_history.append((metric_name, global_val))
        self.did_aggregate.append((metric_name, True))
        self.last_results[metric_name] = global_val

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
        last_global_val = None

        for _ in range(num_rounds):
            self.round_counter += 1
            last_global_val = float(agg_fn())
            self._record_step(local_val, last_global_val, metric_name)

        return last_global_val

    # -------------------- main use cases --------------------

    def fit_mean(self, values: np.ndarray, num_rounds=50) -> float:
        values = self._validate_1d(values)
        n = int(values.size)

        local_sum = float(values.sum())
        local_mean = float(local_sum / n)

        last_global_mean = None
        for _ in range(num_rounds):
            self.round_counter += 1
            last_global_mean = float(self.agg.fed_mean(local_sum, n))
            self._record_step(local_mean, last_global_mean, "mean")

        return last_global_mean

    def fit_metric_average(self, local_metric: float, weight: int, num_rounds=50) -> float:
        local_metric = float(local_metric)
        weight = int(weight)

        if weight <= 0:
            raise ValueError("weight must be > 0")

        last_global_metric = None
        for _ in range(num_rounds):
            self.round_counter += 1
            last_global_metric = float(
                self.agg.fed_weighted_mean_of_metric(local_metric, weight)
            )
            self._record_step(local_metric, last_global_metric, "metric_average")

        return last_global_metric

    def fit_variance(self, values: np.ndarray, num_rounds=50) -> float:
        values = self._validate_1d(values)
        local_var = float(np.var(values))

        def aggregate_variance():
            ex = self._global_mean_from_values(values)
            ex2 = self._global_mean_from_values(values ** 2)
            return float(max(ex2 - ex * ex, 0.0))

        return self._fit_scalar_metric("variance", local_var, aggregate_variance, num_rounds)

    def fit_std(self, values: np.ndarray, num_rounds=50) -> float:
        values = self._validate_1d(values)
        local_std = float(np.std(values))

        def aggregate_std():
            ex = self._global_mean_from_values(values)
            ex2 = self._global_mean_from_values(values ** 2)
            var = max(ex2 - ex * ex, 0.0)
            return float(np.sqrt(var))

        return self._fit_scalar_metric("std", local_std, aggregate_std, num_rounds)

    def fit_rms(self, values: np.ndarray, num_rounds=50) -> float:
        values = self._validate_1d(values)
        local_rms = float(np.sqrt(np.mean(values ** 2)))

        def aggregate_rms():
            ex2 = self._global_mean_from_values(values ** 2)
            return float(np.sqrt(ex2))

        return self._fit_scalar_metric("rms", local_rms, aggregate_rms, num_rounds)

    def fit_covariance(self, x: np.ndarray, y: np.ndarray, num_rounds=50) -> float:
        x, y = self._validate_pair(x, y)
        local_cov = float(np.mean((x - x.mean()) * (y - y.mean())))

        def aggregate_cov():
            ex = self._global_mean_from_values(x)
            ey = self._global_mean_from_values(y)
            exy = self._global_mean_from_values(x * y)
            return float(exy - ex * ey)

        return self._fit_scalar_metric("covariance", local_cov, aggregate_cov, num_rounds)

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

        return self._fit_scalar_metric("pearson", local_pearson, aggregate_pearson, num_rounds)

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

        last_reg = None

        for _ in range(num_rounds):
            self.round_counter += 1

            ex = self._global_mean_from_values(x)
            ey = self._global_mean_from_values(y)
            ex2 = self._global_mean_from_values(x ** 2)
            exy = self._global_mean_from_values(x * y)

            var_x = max(ex2 - ex * ex, 0.0)
            if var_x == 0:
                last_reg = (0.0, float(ey))
            else:
                cov_xy = exy - ex * ey
                slope = float(cov_xy / var_x)
                intercept = float(ey - slope * ex)
                last_reg = (slope, intercept)

            self._record_step(local_val, last_reg, "regression")

        return last_reg

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
        Επιστρέφει το πιο πρόσφατο aggregated αποτέλεσμα για το metric.
        """
        return self.last_results.get(metric_name)