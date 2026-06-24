import numpy as np

from delivery_delay.features.geo import haversine_km


def test_zero_distance():
    assert haversine_km(42.36, -71.06, 42.36, -71.06) == 0.0


def test_one_degree_latitude_is_about_111km():
    d = haversine_km(42.0, -71.0, 43.0, -71.0)
    assert 110 < d < 112


def test_vectorised_matches_scalar():
    lat1 = np.array([42.0, 40.0])
    lon1 = np.array([-71.0, -73.0])
    lat2 = np.array([42.5, 40.5])
    lon2 = np.array([-71.5, -73.5])
    arr = haversine_km(lat1, lon1, lat2, lon2)
    assert arr.shape == (2,)
    assert np.isclose(arr[0], haversine_km(42.0, -71.0, 42.5, -71.5))


def test_distance_is_symmetric():
    a = haversine_km(42.36, -71.06, 42.40, -71.10)
    b = haversine_km(42.40, -71.10, 42.36, -71.06)
    assert np.isclose(a, b)
