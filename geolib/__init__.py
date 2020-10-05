"""
GEOLib Library
"""

__version__ = "0.1.2"

# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

from . import utils
from .models import *

logging.getLogger(__name__).addHandler(NullHandler())
