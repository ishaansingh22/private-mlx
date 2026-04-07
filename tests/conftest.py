"""Pytest configuration: skip heavy tests unless explicitly requested."""

import pytest


def pytest_collection_modifyitems(config, items):
    specified = config.getoption("-m", default="")
    for item in items:
        if "mnist" in item.keywords and "mnist" not in specified:
            item.add_marker(pytest.mark.skip(reason="requires -m mnist"))
        if "lora" in item.keywords and "lora" not in specified:
            item.add_marker(pytest.mark.skip(reason="requires -m lora"))
