import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

from tests_and_experiments.core.partitioned_table import PartitionedPandasTable


class WaterQuality(PartitionedPandasTable):

    def get_dataset(self) -> pd.DataFrame:
        return kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,"adityakadiwal/water-potability","water_potability.csv")





if __name__ == "__main__":
    Data = WaterQuality()
    data = Data.get_dataset()
    print(data.head())