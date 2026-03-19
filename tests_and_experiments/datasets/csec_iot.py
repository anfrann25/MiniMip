import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

from tests_and_experiments.core.partitioned_table import PartitionedPandasTable


class ICS3D(PartitionedPandasTable):

    def get_dataset(self) -> pd.DataFrame:
        return kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "rogernickanaedevha/integrated-cloud-security-3datasets-ics3d",
            "ML-EdgeIIoT-dataset.csv"   # <-- άλλαξέ το αν το πραγματικό filename είναι διαφορετικό
        )


if __name__ == "__main__":
    data = ICS3D().get_dataset()
    print(data.head())
    print(data.shape)
    print(data.columns.tolist())