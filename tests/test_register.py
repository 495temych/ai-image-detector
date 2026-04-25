import pytest
from src.register import should_register


def test_should_register_above_threshold():
    assert should_register(accuracy=0.95, threshold=0.90) is True


def test_should_register_below_threshold():
    assert should_register(accuracy=0.85, threshold=0.90) is False


def test_should_register_at_threshold():
    assert should_register(accuracy=0.90, threshold=0.90) is True


def test_should_register_zero_accuracy():
    assert should_register(accuracy=0.0, threshold=0.90) is False
