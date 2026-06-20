import numpy as np

from elrpy.rays import (
    evaluate_categorical_loss_rays,
    make_ray_directions,
    summarize_ray_losses,
)


def test_make_ray_directions_returns_unit_array():
    directions = make_ray_directions(
        3,
        n_random=2,
        seed=0,
        include_antipodes=True,
        include_coordinates=True,
        gradient=np.array([1.0, 0.0, 0.0]),
    )

    assert directions.shape == (12, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    np.testing.assert_allclose(directions[0], -directions[2])
    np.testing.assert_allclose(directions[-2], np.array([-1.0, 0.0, 0.0]))
    np.testing.assert_allclose(directions[-1], np.array([1.0, 0.0, 0.0]))


def test_summarize_ray_losses_separates_failures():
    radii = np.array([0.0, 1.0, 2.0])
    losses = np.array(
        [
            [1.0, 1.1, 1.2],
            [1.0, 0.9, 1.3],
            [1.0, 1.5, 1.4],
        ]
    )
    directions = np.eye(3)

    summary = summarize_ray_losses(
        losses,
        radii,
        directions=directions,
        gradient=np.array([0.2, -0.3, 0.1]),
        tolerance=1e-8,
        top_k=2,
    )

    assert summary["lower_than_center_directions"] == 1
    assert summary["non_monotone_directions"] == 2
    assert summary["top_lower_than_center"][0]["direction_index"] == 1
    assert summary["top_non_monotone"][0]["direction_index"] == 1
    assert summary["negative_initial_derivative_directions"] == 1
    assert summary["top_negative_initial_derivative"][0]["direction_index"] == 1


def test_evaluate_categorical_loss_rays_shape_and_center_agreement():
    X = np.array(
        [
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.5],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    G = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    Y = np.array(
        [
            [1.0, 0.7, 0.3],
            [0.4, 1.2, 0.4],
        ],
        dtype=np.float32,
    )
    center = np.zeros(4, dtype=np.float32)
    directions = np.eye(4, dtype=np.float32)[:2]
    radii = np.array([0.0, 0.25, 0.5])

    losses = evaluate_categorical_loss_rays(
        X,
        Y,
        G,
        center,
        directions,
        radii,
        n_outcomes=3,
        batch_size=2,
    )

    assert losses.shape == (2, 3)
    np.testing.assert_allclose(losses[0, 0], losses[1, 0], rtol=1e-6)
    assert np.isfinite(losses).all()
