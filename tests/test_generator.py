from delivery_delay.data.generator import (
    CANONICAL_COLUMNS,
    add_targets,
    generate_orders,
)


def test_schema_and_size(cfg):
    df = generate_orders(cfg, n_orders=1500, seed=1)
    assert list(df.columns) == CANONICAL_COLUMNS
    assert len(df) == 1500


def test_determinism(cfg):
    a = generate_orders(cfg, n_orders=1000, seed=42)
    b = generate_orders(cfg, n_orders=1000, seed=42)
    assert a.equals(b)


def test_targets_and_delay_rate(cfg):
    df = add_targets(generate_orders(cfg, n_orders=4000, seed=3), cfg)
    assert {"delay_minutes", "is_delayed"}.issubset(df.columns)
    # A useful classification problem should not be all-or-nothing.
    rate = df["is_delayed"].mean()
    assert 0.02 < rate < 0.9


def test_actual_minutes_positive(cfg):
    df = generate_orders(cfg, n_orders=1000, seed=5)
    assert (df["actual_minutes"] > 0).all()
