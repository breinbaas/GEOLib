from pydantic import BaseModel as DataModel
from enum import IntEnum
from pydantic import PositiveFloat, constr, confloat
from typing import Optional
from .settings import Side
from .internal import (
    Anchor as InternalAnchor,
    Strut as InternalStrut,
    Support as InternalSupport,
)


class Anchor(DataModel):
    """Anchor. This option is not available for SinglePileModelType.

    Args:
        name: Name of the anchor
        level: Level of the anchor, or z-coordinate [m]
        e_modulus: E-modulus [kN/m^2]
        cross_section: Cross section [m^2/m']
        wall_height_kranz: Height of the wall [Kranz] [m]
        length: Length [m]
        angle: Angle [deg]
        yield_force: Yield force [kN/m']
        side: Side of the anchor [Side]
    """

    name: constr(min_length=1, max_length=50)
    level: float
    e_modulus: Optional[PositiveFloat] = None
    cross_section: Optional[PositiveFloat] = None
    wall_height_kranz: Optional[PositiveFloat] = None
    length: Optional[PositiveFloat] = None
    angle: Optional[float] = None
    side: Side = Side.RIGHT
    yield_force: Optional[PositiveFloat] = None

    def to_internal(self) -> InternalAnchor:
        return InternalAnchor(**self.dict(exclude_none=True))


class Strut(DataModel):
    """Strut. This option is not available for SinglePileModelType.

    Args:
        name: Name of the strut
        level: Level of the strut, or z-coordinate [m]
        e_modulus: E-modulus [kN/m^2]
        cross_section: Cross section [m^2/m']
        length: Length [m]
        angle: Angle [deg]
        buckling_force: Buckling force of the strut [kN/m']
        side: Side of the strut [Side]
        pre_compression: Pre-compressions [kN/m']
    """

    name: constr(min_length=1, max_length=50)
    level: float
    e_modulus: Optional[PositiveFloat] = None
    cross_section: Optional[PositiveFloat] = None
    length: Optional[PositiveFloat] = None
    angle: Optional[PositiveFloat] = None
    buckling_force: Optional[PositiveFloat] = None
    side: Side = Side.RIGHT
    pre_compression: Optional[PositiveFloat] = None

    def to_internal(self) -> InternalStrut:
        return InternalStrut(**self.dict(exclude_none=True, exclude={"pre_compression"}))


class SupportType(IntEnum):
    TRANSLATION = 1
    ROTATION = 2
    TRANSLATION_AND_ROTATION = 3


class SpringSupport(DataModel):
    """Spring support."""

    name: constr(min_length=1, max_length=50)
    level: float
    rotational_stiffness: confloat(ge=0)
    translational_stiffness: confloat(ge=0)

    def to_internal(self) -> InternalSupport:
        return InternalSupport(**self.dict())


class RigidSupport(DataModel):
    """Rigid support."""

    name: constr(min_length=1, max_length=50)
    level: float
    support_type: SupportType

    def to_internal(self) -> InternalSupport:

        rotational_stiffness, translational_stiffness = {
            SupportType.TRANSLATION.value: (0, 1),
            SupportType.ROTATION.value: (1, 0),
            SupportType.TRANSLATION_AND_ROTATION.value: (1, 1),
        }[self.support_type.value]

        return InternalSupport(
            name=self.name,
            level=self.level,
            rotational_stiffness=rotational_stiffness,
            translational_stiffness=translational_stiffness,
        )
