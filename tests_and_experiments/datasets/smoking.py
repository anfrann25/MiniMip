import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

from tests_and_experiments.core.partitioned_table import PartitionedPandasTable


class SmokingDataset(PartitionedPandasTable):

    def get_dataset(self) -> pd.DataFrame:
        # Κατέβασε το dataset (επιστρέφει path)
        path = kagglehub.dataset_download(
            "kukuroo3/body-signal-of-smoking"
        )

        # Φόρτωσε το CSV με σωστό encoding
        df = pd.read_csv(
            f"{path}/smoking.csv",
            encoding="latin1"  # <-- ΤΟ ΚΛΕΙΔΙ
        )

        return df




if __name__ == "__main__":
    ds = SmokingDataset()
    df = ds.get_dataset()
    print(df.head())
    print(df.info())