import numpy as np
from sklearn.linear_model import LogisticRegression
from library.core.statistical_model import StatisticalModel
from library.utils.numpy_aggregator import NumpyAggregator
from mini_mip_system.client.grpc_agg_client import AggregationClientInterface
from sklearn.metrics import accuracy_score


class FederatedLogisticRegressionClientCached(StatisticalModel):
    """
    Cached client optimized to reduce server calls and save time.
    Aggregates only every `aggregation_interval` rounds.
    """
    def __init__(self, client: AggregationClientInterface, aggregation_interval=3, model_params=None):
        super().__init__(client)
        self.agg = NumpyAggregator(self.client)
        self.aggregation_interval = aggregation_interval
        self.accuracy_history = []  # για accuracy ανά epoch

        self.model_params = model_params or {
            'solver': 'saga', 'penalty': 'l2', 'fit_intercept': True,
            'max_iter': 100, 'warm_start': True
        }
        self.model = LogisticRegression(**self.model_params)

        self.cache = {
            "coef": None,
            "intercept": None
        }

        self.round_counter = 0

    #LocalUpdate
    def _local_update(self, X, y):
        if hasattr(self.model, 'partial_fit'):
            # partial_fit υπάρχει
            if self.round_counter == 0:
                self.model.partial_fit(X, y, classes=np.unique(y))
            else:
                self.model.partial_fit(X, y) #μαθαίνει απο τα τοπικά δεδομένα
        else:
            # fallback σε κανονικό fit
            self.model.fit(X, y)

    #CacheUpdate
    def _update_cache(self, coef_new, intercept_new):
        # Save new weights with timestamp
        self.cache["coef"] = (coef_new.copy(), self.round_counter)
        self.cache["intercept"] = (intercept_new.copy(), self.round_counter)

        # Remove stale cache entries
        for key in ["coef", "intercept"]:
            entry = self.cache[key]
            if entry is None:
                continue

            _, t_entry = entry

            if (self.round_counter - t_entry) > self.tau_max:
                # Stale → delete
                self.cache[key] = None

    # =============================================================
    #   CACHED AGGREGATION
    # =============================================================
    def _cached_fed_avg(self, local_w, cached_entry):
        """
        Weighted avg between:
           - local model weights
           - cached remote weights (if not stale)
        """

        if cached_entry is None:
            return local_w     # No cached values

        cached_w, _ = cached_entry

        # 2-way average: 0.5 * local + 0.5 * cached
        return 0.5 * local_w + 0.5 * cached_w

    #mainloop!
    def fit(self, X: np.ndarray, y: np.ndarray, num_epochs: int = 100, X_val=None, y_val=None):
        for _ in range(num_epochs):
            self.round_counter += 1
            print(f"[Client CachedFast] Epoch {self.round_counter}/{num_epochs}")

            # 1) Local update
            self._local_update(X, y)

            # 2) Aggregation only every `aggregation_interval` rounds
            if self.round_counter % self.aggregation_interval == 0:
                new_coef = self.agg.fed_weighted_avg(self.model.coef_, X.shape[0])
                new_intercept = self.agg.fed_weighted_avg(self.model.intercept_, X.shape[0])
                # Update cache
                self.cache["coef"] = new_coef.copy()
                self.cache["intercept"] = new_intercept.copy()

            # 3) Use cached weights for intermediate rounds
            if self.cache["coef"] is not None:
                self.model.coef_ = self.cache["coef"].copy()
            if self.cache["intercept"] is not None:
                self.model.intercept_ = self.cache["intercept"].copy()

            # -----------------------------
            # 4) Υπολογισμός accuracy αν υπάρχει validation set
            if X_val is not None and y_val is not None:
                y_pred = self.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                self.accuracy_history.append(acc)



    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)
