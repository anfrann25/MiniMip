import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

from tests_and_experiments.core.partitioned_table import PartitionedPandasTable


class LungCancer(PartitionedPandasTable):

    def get_dataset(self) -> pd.DataFrame:
        return kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,"khwaishsaxena/lung-cancer-dataset","Lung Cancer.csv")


if __name__ == "__main__":
    data = LungCancer().get_dataset()
    print(data.head())



