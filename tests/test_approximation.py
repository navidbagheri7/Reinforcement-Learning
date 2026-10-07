import numpy as np

from decision_lab.approximation import Tiles


def test_tile_features_and_action_separation():
    tiles = Tiles()
    for state in [[-1.2, -0.07], [0.6, 0.07], [-0.5, 0]]:
        groups = [tiles.indices(state, a) for a in range(3)]
        assert all(len(np.unique(g)) == 8 for g in groups)
        assert min(np.concatenate(groups)) >= 0
        assert max(np.concatenate(groups)) < tiles.size
        assert not np.intersect1d(groups[0], groups[1]).size
        weights = np.zeros(tiles.size)
        weights[groups[1]] = 2
        assert np.array_equal(tiles.q(weights, state), [0, 16, 0])
