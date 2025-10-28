from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from lcpfn.models import distributions


def test_get_bucket_limits_recovers_monotonic_borders_from_repeated_values():
    ys = torch.cat([torch.zeros(5_000), torch.ones(5_000)])

    bucket_limits = distributions.get_bucket_limits(256, ys=ys)

    diffs = bucket_limits[1:] - bucket_limits[:-1]
    assert torch.all(
        diffs > 0
    ), "Bucket limits should be strictly increasing after sanitization"

    dist = distributions.BarDistribution(bucket_limits)
    assert torch.all(dist.bucket_widths > 0)


def test_get_bucket_limits_handles_constant_targets():
    ys = torch.zeros(1_000)

    bucket_limits = distributions.get_bucket_limits(128, ys=ys)

    diffs = bucket_limits[1:] - bucket_limits[:-1]
    assert torch.all(
        diffs > 0
    ), "Constant targets should still result in valid bucket limits"

    dist = distributions.BarDistribution(bucket_limits)
    assert torch.all(dist.bucket_widths > 0)
