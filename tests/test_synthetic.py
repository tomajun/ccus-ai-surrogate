"""
test_synthetic.py —— 合成数据与数据集的基本健全性测试
=====================================================
运行：pytest tests/
（只需 numpy，不依赖 torch，保证数据管线随时可验证）
"""
import numpy as np

from sim.synthetic_field import generate_plume, risk_score


def test_shape_and_range():
    p = dict(injection_rate=1.5, caprock_perm=0.8, thickness=30.0)
    f = generate_plume(p, grid_n=48, n_steps=10, seed=0)
    assert f.shape == (10, 48, 48)
    assert f.min() >= 0.0 and f.max() <= 0.85 + 1e-6


def test_risk_label_in_range():
    for seed in range(20):
        p = dict(injection_rate=1.0, caprock_perm=0.5, thickness=30.0)
        f = generate_plume(p, grid_n=48, n_steps=10, seed=seed)
        _, lab = risk_score(p, f[-1])
        assert lab in (0, 1, 2)


def test_high_perm_higher_risk():
    lo = dict(injection_rate=2.0, caprock_perm=0.1, thickness=40.0)
    hi = dict(injection_rate=2.0, caprock_perm=1.0, thickness=40.0)
    _, lab_lo = risk_score(lo, generate_plume(lo, seed=1)[-1])
    _, lab_hi = risk_score(hi, generate_plume(hi, seed=1)[-1])
    # 同注入速率下，盖层渗透性越高，风险不应更低
    assert lab_hi >= lab_lo
