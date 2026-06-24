from delivery_delay.data.generator import add_targets, generate_orders
from delivery_delay.features.build import build_features, build_xy, feature_columns


def test_feature_columns_match_builder(cfg):
    df = generate_orders(cfg, n_orders=200, seed=2)
    feats = build_features(df, cfg)
    assert set(feats.columns) == set(feature_columns(cfg))


def test_build_xy_shapes(cfg):
    df = add_targets(generate_orders(cfg, n_orders=300, seed=2), cfg)
    X, y_eta, y_delay = build_xy(df, cfg)
    assert len(X) == len(df) == len(y_eta) == len(y_delay)
    assert list(X.columns) == feature_columns(cfg)


def test_single_row_parity(cfg):
    """One row scored alone must yield the same features as in a batch."""
    df = generate_orders(cfg, n_orders=50, seed=11)
    batch = build_features(df, cfg).reindex(columns=feature_columns(cfg), fill_value=0.0)
    single = build_features(df.iloc[[0]], cfg).reindex(columns=feature_columns(cfg), fill_value=0.0)
    assert single.iloc[0].round(6).equals(batch.iloc[0].round(6))


def test_no_nans(cfg):
    df = generate_orders(cfg, n_orders=200, seed=2)
    feats = build_features(df, cfg)
    assert not feats.isna().any().any()
