"""
Common utility methods used within GEOLib.
"""

import csv
import logging
import re
from collections import namedtuple
from pathlib import Path
from typing import Any, List, Tuple
from shapely import LineString, MultiPoint, Point

from pydantic import validator

_CAMEL_TO_SNAKE_PATTERN = re.compile(r"(?<!^)(?=[A-Z])")

logger = logging.getLogger(__name__)


def camel_to_snake(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return _CAMEL_TO_SNAKE_PATTERN.sub("_", name).lower()


def snake_to_camel(name: str) -> str:
    return "".join(word.title() for word in name.split("_"))


def csv_as_namedtuples(fn: Path, delimiter=";"):
    with open(fn, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = namedtuple("Header", next(reader))
        for data in map(header._make, reader):
            yield data


def make_newline_validator(*field_name: str, req_newlines: int = 2):
    """Get a dynamic validator for ensuring a set number of lines."""

    def field_must_contain_newlines(v: str):
        newlines = v.count("\n")
        if newlines < req_newlines:
            logger.warning(
                f"Added {req_newlines - newlines} lines to run_identification."
            )
            v += (req_newlines - newlines) * "\n"
        elif newlines > req_newlines:
            logger.warning(
                f"More than {req_newlines+1} lines in run_identification will be ignored in the GUI."
            )
        return v

    return validator(*field_name, allow_reuse=True)(field_must_contain_newlines)


def polyline_polyline_intersections(
    points_line1: List[Tuple[float, float]],
    points_line2: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    result = []
    ls1 = LineString(points_line1)
    ls2 = LineString(points_line2)
    intersections = ls1.intersection(ls2)

    if intersections.is_empty:
        final_result = []
    elif type(intersections) == MultiPoint:
        result = [(g.x, g.y) for g in intersections.geoms]
    elif type(intersections) == Point:
        x, y = intersections.coords.xy
        result.append((x[0], y[0]))
    elif intersections.is_empty:
        return []
    else:
        raise ValueError(f"Unimplemented intersection type '{type(intersections)}'")

    # do not include points that are on line1 or line2
    final_result = [p for p in result if not p in points_line1 or p in points_line2]

    if len(final_result) == 0:
        return []

    return sorted(final_result, key=lambda x: x[0])
