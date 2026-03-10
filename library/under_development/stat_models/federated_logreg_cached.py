import numpy as np
from sklearn.linear_model import LogisticRegression
from library.core.statistical_model import StatisticalModel
from library.utils.numpy_aggregator import NumpyAggregator
from mini_mip_system.client.grpc_agg_client import AggregationClientInterface
from sklearn.metrics import accuracy_score

import numpy as np

def _l2_norm(a):
    return float(np.linalg.norm(a))

class FederatedLogisticRegressionClientCached(StatisticalModel):
    def __init__(
        self,
        client: AggregationClientInterface,
        aggregation_interval=3,     # μπορείς να το κρατήσεις ως "fallback"
        warmup_rounds=5,
        tau_max=10,                 # max epochs που επιτρέπεις να μείνει stale το cached global
        eps_w=1e-3,                 # variation-gap threshold (Sancus Def.4 analog)
        mix_local=0.7,              # πόσο κρατάς local
        mix_cached=0.3,             # πόσο τραβάς προς cached global
        model_params=None

    ):
        super().__init__(client)
        self.did_aggregate_history = []
        self.aggregate_count = 0
        self.skip_count = 0
        self.agg = NumpyAggregator(self.client)

        self.aggregation_interval = aggregation_interval
        self.warmup_rounds = warmup_rounds
        self.tau_max = tau_max
        self.eps_w = eps_w
        self.mix_local = mix_local
        self.mix_cached = mix_cached

        self.accuracy_history = []
        self.model_params = model_params or {
            "solver": "saga", "penalty": "l2", "fit_intercept": True,
            "max_iter": 100, "warm_start": True
        }
        self.model = LogisticRegression(**self.model_params)

        # cache entries are always (value, round)
        self.cache = {"coef": None, "intercept": None}
        self.round_counter = 0

        # keep previous weights to compute variation-gap
        self.prev_coef = None
        self.prev_intercept = None

    def _local_update(self, X, y):
        if hasattr(self.model, "partial_fit"):
            if self.round_counter == 0:
                self.model.partial_fit(X, y, classes=np.unique(y))
            else:
                self.model.partial_fit(X, y)
        else:
            self.model.fit(X, y)

    def _update_cache(self, coef_new, intercept_new):
        self.cache["coef"] = (coef_new.copy(), self.round_counter)
        self.cache["intercept"] = (intercept_new.copy(), self.round_counter)

        # evict too-stale cache
        for k in ["coef", "intercept"]:
            entry = self.cache[k]
            if entry is None:
                continue
            _, t_entry = entry
            if (self.round_counter - t_entry) > self.tau_max:
                self.cache[k] = None

    def _should_aggregate(self):
        """
        Sancus-style staleness check:
        - warmup: always aggregate
        - if cache is too old: aggregate
        - if local weights changed a lot: aggregate
        - else skip
        """
        if self.round_counter <= self.warmup_rounds:
            return True

        # if no cache, we must aggregate at least once
        if self.cache["coef"] is None or self.cache["intercept"] is None:
            return True

        # tau-based bound (epoch gap)
        _, t_coef = self.cache["coef"]
        if (self.round_counter - t_coef) > self.tau_max:
            return True

        # variation-gap bound (Def.4 analog)
        if self.prev_coef is None or self.prev_intercept is None:
            return True

        dcoef = _l2_norm(self.model.coef_ - self.prev_coef)
        dint = _l2_norm(self.model.intercept_ - self.prev_intercept)

        # aggregate only if change is "large"
        return (dcoef + dint) > self.eps_w

    def _mix_with_cached(self):
        # between aggregations: mix local with cached global
        if self.cache["coef"] is not None:
            cached_coef, _ = self.cache["coef"]
            self.model.coef_ = self.mix_local * self.model.coef_ + self.mix_cached * cached_coef

        if self.cache["intercept"] is not None:
            cached_int, _ = self.cache["intercept"]
            self.model.intercept_ = self.mix_local * self.model.intercept_ + self.mix_cached * cached_int

    def fit(self, X, y, num_epochs=100, X_val=None, y_val=None):
        for _ in range(num_epochs):
            self.round_counter += 1
            print(f"[Client fedChSa-like] Epoch {self.round_counter}/{num_epochs}")

            # save previous for variation-gap measurement
            if hasattr(self.model, "coef_"):
                self.prev_coef = self.model.coef_.copy()
                self.prev_intercept = self.model.intercept_.copy()

            # 1) local update
            self._local_update(X, y)

            did_aggregate = False
            # 2) staleness-aware skip-broadcast
            if self._should_aggregate():
                global_coef = self.agg.fed_weighted_avg(self.model.coef_, X.shape[0])
                global_intercept = self.agg.fed_weighted_avg(self.model.intercept_, X.shape[0])

                # update cache + apply
                self._update_cache(global_coef, global_intercept)
                self.model.coef_ = global_coef.copy()
                self.model.intercept_ = global_intercept.copy()
                did_aggregate = True
            else:
                self._mix_with_cached()

            self.did_aggregate_history.append(did_aggregate)

            if did_aggregate:
                self.aggregate_count += 1
            else:
                self.skip_count += 1

            # optional fallback: ensure periodic aggregation
            # (π.χ. για να μη “κολλήσει”)
            if self.round_counter % self.aggregation_interval == 0:
                if self.cache["coef"] is None:
                    global_coef = self.agg.fed_weighted_avg(self.model.coef_, X.shape[0])
                    global_intercept = self.agg.fed_weighted_avg(self.model.intercept_, X.shape[0])
                    self._update_cache(global_coef, global_intercept)
                    self.model.coef_ = global_coef.copy()
                    self.model.intercept_ = global_intercept.copy()

            # 3) accuracy
            if X_val is not None:
                y_pred = self.predict(X_val)
                self.accuracy_history.append(accuracy_score(y_val, y_pred))

    def predict(self, x, *args, **kwargs):
        return self.model.predict(x)

    def predict_proba(self, x, *args, **kwargs):
        return self.model.predict_proba(x)

