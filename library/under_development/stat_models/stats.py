import numpy as np
import math


# =========================================================
# SERVER
# =========================================================
class AggregationServer:
    def __init__(self):
        self.client_data = {}

    def register_client_data(self, client_id, data):
        self.client_data[client_id] = np.array(data)

    def global_count(self):
        total = 0
        for arr in self.client_data.values():
            total += arr.shape[0]
        return total

    def global_sum_scalar(self):
        total = 0.0
        for arr in self.client_data.values():
            total += np.sum(arr)
        return total

    def global_avg_scalar(self):
        total_count = self.global_count()
        if total_count == 0:
            raise ValueError("No data in federation")
        return self.global_sum_scalar() / total_count

    def global_min_scalar(self):
        if not self.client_data:
            raise ValueError("No clients registered")
        mins = [np.min(arr) for arr in self.client_data.values()]
        return min(mins)

    def global_max_scalar(self):
        if not self.client_data:
            raise ValueError("No clients registered")
        maxs = [np.max(arr) for arr in self.client_data.values()]
        return max(maxs)

    def global_union(self):
        if not self.client_data:
            return np.array([])
        all_values = np.concatenate(list(self.client_data.values()))
        return np.unique(all_values)

    def global_variance(self, ddof=0):
        n = self.global_count()
        if n == 0:
            raise ValueError("No data in federation")
        if n <= ddof:
            raise ValueError(f"Not enough data points for ddof={ddof}")

        mean = self.global_avg_scalar()
        sq_diff_sum = 0.0
        for arr in self.client_data.values():
            sq_diff_sum += np.sum((arr - mean) ** 2)

        return sq_diff_sum / (n - ddof)

    def global_std(self, ddof=0):
        return np.sqrt(self.global_variance(ddof=ddof))

    def global_covariance(self, x_map, y_map, ddof=0):
        # x_map, y_map: dict {client_id: np.array}
        n = sum(len(v) for v in x_map.values())
        if n == 0:
            raise ValueError("Empty federated data")
        if n <= ddof:
            raise ValueError(f"Not enough data points for ddof={ddof}")

        all_x = np.concatenate([x_map[cid] for cid in x_map])
        all_y = np.concatenate([y_map[cid] for cid in y_map])

        mean_x = np.mean(all_x)
        mean_y = np.mean(all_y)

        return np.sum((all_x - mean_x) * (all_y - mean_y)) / (n - ddof)

    def global_pearson(self, x_map, y_map):
        all_x = np.concatenate([x_map[cid] for cid in x_map])
        all_y = np.concatenate([y_map[cid] for cid in y_map])

        n = len(all_x)
        if n == 0:
            raise ValueError("Empty federated data")
        if n <= 1:
            return 0

        mean_x = np.mean(all_x)
        mean_y = np.mean(all_y)

        cov = np.sum((all_x - mean_x) * (all_y - mean_y)) / (n - 1)
        var_x = np.sum((all_x - mean_x) ** 2) / (n - 1)
        var_y = np.sum((all_y - mean_y) ** 2) / (n - 1)

        if var_x <= 0 or var_y <= 0:
            return 0

        return cov / (math.sqrt(var_x) * math.sqrt(var_y))


# =========================================================
# CLIENT
# =========================================================
class AggregationClient:
    def __init__(self, client_id, server, local_data):
        self.client_id = client_id
        self.server = server
        self.local_data = np.array(local_data, dtype=float)
        self.server.register_client_data(client_id, self.local_data)

    def get_local_data(self):
        return self.local_data

    def update_local_data(self, data):
        self.local_data = np.array(data, dtype=float)
        self.server.register_client_data(self.client_id, self.local_data)


# =========================================================
# FEDERATED STATISTICS USING SERVER
# =========================================================
class FederatedStatisticsDemo:
    def __init__(self, server: AggregationServer):
        self.server = server

    def fed_count(self):
        return self.server.global_count()

    def mean(self):
        return self.server.global_avg_scalar()

    def variance(self, ddof=0):
        return self.server.global_variance(ddof=ddof)

    def standard_deviation(self, ddof=0):
        return self.server.global_std(ddof=ddof)

    def sum_of_squares(self):
        total = 0.0
        for arr in self.server.client_data.values():
            total += np.sum(arr ** 2)
        return total

    def data_range(self):
        return self.server.global_max_scalar() - self.server.global_min_scalar()

    def coefficient_of_variation(self):
        avg = self.mean()
        if avg == 0:
            return 0
        return self.standard_deviation() / avg

    def mean_absolute_deviation(self):
        mean = self.mean()
        n = self.fed_count()
        if n == 0:
            return 0
        total_abs = 0.0
        for arr in self.server.client_data.values():
            total_abs += np.sum(np.abs(arr - mean))
        return total_abs / n

    def root_mean_square(self):
        n = self.fed_count()
        if n == 0:
            return 0
        total_sq = 0.0
        for arr in self.server.client_data.values():
            total_sq += np.sum(arr ** 2)
        return np.sqrt(total_sq / n)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    server = AggregationServer()

    client1 = AggregationClient("client1", server, [1, 2, 3])
    client2 = AggregationClient("client2", server, [4, 5, 6])
    client3 = AggregationClient("client3", server, [7, 8, 9])

    stats = FederatedStatisticsDemo(server)

    print("----- CLIENT DATA -----")
    for cid, arr in server.client_data.items():
        print(cid, "->", arr)

    print("\n----- FEDERATED RESULTS -----")
    print("count:", stats.fed_count())
    print("mean:", stats.mean())
    print("variance (population):", stats.variance(ddof=0))
    print("variance (sample):", stats.variance(ddof=1))
    print("std (population):", stats.standard_deviation(ddof=0))
    print("std (sample):", stats.standard_deviation(ddof=1))
    print("sum_of_squares:", stats.sum_of_squares())
    print("range:", stats.data_range())
    print("coefficient_of_variation:", stats.coefficient_of_variation())
    print("mean_absolute_deviation:", stats.mean_absolute_deviation())
    print("root_mean_square:", stats.root_mean_square())
    print("union:", server.global_union())

    # Παράδειγμα για covariance / pearson με 3 clients
    x_map = {
        "client1": np.array([1, 2, 3], dtype=float),
        "client2": np.array([4, 5, 6], dtype=float),
        "client3": np.array([7, 8, 9], dtype=float),
    }

    y_map = {
        "client1": np.array([2, 4, 6], dtype=float),
        "client2": np.array([8, 10, 12], dtype=float),
        "client3": np.array([14, 16, 18], dtype=float),
    }

    print("\n----- FEDERATED RELATION x,y -----")
    print("covariance population:", server.global_covariance(x_map, y_map, ddof=0))
    print("covariance sample:", server.global_covariance(x_map, y_map, ddof=1))
    print("pearson:", server.global_pearson(x_map, y_map))