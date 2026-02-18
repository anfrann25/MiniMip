import numpy as np
from sklearn.linear_model import LogisticRegression

from library.core.statistical_model import StatisticalModel
from library.utils.numpy_aggregator import NumpyAggregator
from mini_mip_system.client.grpc_agg_client import AggregationClientInterface
from sklearn.metrics import accuracy_score


class FederatedLogisticRegressionClientSaSo(StatisticalModel):

    def __init__(self, client: AggregationClientInterface, model_params=None):

        super().__init__(client)
        self.agg = NumpyAggregator(self.client)
        self.accuracy_history = []  # για accuracy ανά epoch
        self.model_params = model_params or {
            'solver': 'saga', 'penalty': 'l2', 'fit_intercept': True,
            'max_iter': 100, 'warm_start': True
        }
        self.model = LogisticRegression(**self.model_params)

    def fit(self, X: np.ndarray, y: np.ndarray, num_epochs: int = 100, X_val=None, y_val=None):

        """
        Federated training loop. Performs one epoch of local training followed
        by federated aggregation after each epoch.

        Args:
            num_epochs (int): Number of global training epochs (rounds of aggregation).
        """
        n_samples = X.shape[0]

        for epoch in range(num_epochs):
            print(f"[Client] Federated Epoch {epoch + 1}/{num_epochs}")

            # One local epoch: partial_fit for better control
            if hasattr(self.model, 'partial_fit'):
                # For binary classification, need to specify classes on first call
                if epoch == 0:
                    self.model.partial_fit(X, y, classes=np.unique(y))
                else:
                    self.model.partial_fit(X, y)
            else:
                # Fallback to full fit, warm_start avoids reinitializing weights
                self.model.fit(X, y)

            # Extract weights (coef_ and intercept_)
            self.model.coef_ = self.agg.fed_weighted_avg(self.model.coef_, X.shape[0])
            self.model.intercept_ = self.agg.fed_weighted_avg(self.model.intercept_, X.shape[0])

            if X_val is not None:
                y_pred = self.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                self.accuracy_history.append(acc)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)
