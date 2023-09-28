import pytest

from geolib import __version__

from .context import geolib

version = "1.3.2"


@pytest.mark.systemtest
def test_version():
    assert __version__ == version
