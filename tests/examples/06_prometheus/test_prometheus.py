"""Tests for examples in 06_prometheus."""

import runpy
import os
import pytest
from glob import glob

from graphnet.constants import GRAPHNET_ROOT_DIR

EXAMPLE_PATH = os.path.join(GRAPHNET_ROOT_DIR, "examples/06_prometheus")
examples = glob(EXAMPLE_PATH + "/*.py")


@pytest.mark.parametrize("example", examples)
def test_script_execution(example: str) -> None:
    """Test function that executes example."""
    runpy.run_path(os.path.join(EXAMPLE_PATH, example))
