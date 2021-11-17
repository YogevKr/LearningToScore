import unittest

import pytorch_lightning as pl

from learning_to_score.datasets import ClustersDataset


class TestClustersDataset(unittest.TestCase):
    def setUp(self) -> None:
        pl.seed_everything(0)

        input_dim = 200
        num_of_clusters = 10
        number_of_samples_in_cluster = 1000
        cluster_dim_mean = 10
        mean = 0
        variance = 5

        self.dataset = ClustersDataset(
            num_of_clusters=num_of_clusters,
            number_of_samples_in_cluster=number_of_samples_in_cluster,
            dim=input_dim,
            cluster_dim_mean=cluster_dim_mean,
            mean=mean,
            variance=variance,
        )
        self.dataset.setup()

    def test_len(self):
        assert (
            self.dataset.X.shape[0]
            == self.dataset.S.shape[0]
            == self.dataset.Y.shape[0]
        )

    def test_side_information(self):
        for d in self.dataset:
            (
                _,
                _,
                _,
                side_information_a,
                side_information_p,
                side_information_n,
                y_a,
                y_p,
                y_n,
            ) = d

            assert (side_information_a == y_a).all()
            assert (side_information_p == y_p).all()
            assert (side_information_n == y_n).all()


if __name__ == "__main__":
    unittest.main()
