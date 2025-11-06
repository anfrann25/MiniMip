from sklearn.neighbors import NearestNeighbors

from library.under_development.causal.propensity_score import PropensityScore
from library.under_development.mixed_effects.centralized_mixed_effect import mixed_effect
from library.under_development.mixed_effects.federated_mixed_effect import FederatedMixedEffect
from tests_and_experiments.core.test_template import FederationTestTemplate
import pandas as pd

from sklearn.linear_model import LogisticRegression

class Test_FederatedMixedEffect(FederationTestTemplate):

    def __init__(self, client_id, client_count, *, patients, covariates,center,outcome):
        self.covariates = covariates
        self.center = center
        self.outcome = outcome
        super().__init__(client_id, client_count, dataset=patients)


    def federated_computation(self, local_dataset):
        f_mff = FederatedMixedEffect(self.client)
        return f_mff.compute(local_dataset,covariates=self.covariates,center=self.center,outcome=self.outcome)

    def centralized_computation(self, centralized_dataset):
        return mixed_effect(centralized_dataset,covariates=self.covariates,center=self.center,outcome=self.outcome)

    def compare(self, federated_output, global_output):
        sxx,syy = federated_output
        sxx_gl, syy_gl = global_output
        print(syy)
        print(syy_gl)
        # print(sxx - sxx_gl)



