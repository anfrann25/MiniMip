from tests_and_experiments.datasets.iris import IrisDataset
from tests_and_experiments.library_tests.linear_regression.linear_regression_test import LinearRegressionTest
from tests_and_experiments.datasets.insuranse import InsuranceDataset
from tests_and_experiments.library_tests.logistic_regression.logistic_regression_tests import LogisticRegressionTest


LogisticRegressionTest(0, 2,
                           dataset=IrisDataset(),
                           features=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
                           target='target',
                           operation_id=3)

