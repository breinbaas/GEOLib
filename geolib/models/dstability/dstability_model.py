import abc
from enum import Enum
from pathlib import Path
import logging
from math import isnan, exp
from typing import BinaryIO, List, Optional, Set, Type, Union, Dict, Tuple

import matplotlib.pyplot as plt
from pydantic import DirectoryPath, FilePath
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize
from shapely.validation import make_valid
import subprocess

from geolib.geometry import Point
from geolib.models import BaseModel
from geolib.soils import Soil

from .analysis import DStabilityAnalysisMethod
from .dstability_parserprovider import DStabilityParserProvider
from .internal import (
    AnalysisType,
    BishopSlipCircleResult,
    CalculationSettings,
    CharacteristicPointEnum,
    DStabilityResult,
    DStabilityStructure,
    EmbankmentSoilScenarioEnum,
    Geometry,
    PersistableHeadLine,
    PersistableLayer,
    PersistablePoint,
    PersistableSoil,
    PersistableStateCorrelation,
    PersistableExcavation,
    Scenario,
    SoilCollection,
    SoilCorrelation,
    SoilLayerCollection,
    SoilVisualisation,
    SpencerSlipPlaneResult,
    UpliftVanSlipCircleResult,
    Waternet,
    WaternetCreatorSettings,
)
from .loads import Consolidation, DStabilityLoad
from .reinforcements import DStabilityReinforcement
from .serializer import DStabilityInputSerializer, DStabilityInputZipSerializer
from .states import DStabilityStateLinePoint, DStabilityStatePoint
from ..meta import MetaData
from ...errors import CalculationError, WaternetCreatorError
from ...utils import polyline_polyline_intersections, top_of_polygon, bottom_of_polygon
from ...const import UNIT_WEIGHT_WATER

logger = logging.getLogger(__name__)
meta = MetaData()


class DStabilityCalculationType(Enum):
    """Set Type of Calculation."""

    BoundarySearch = 1
    SingleCalc = 2


class DStabilityCalculationModel(Enum):
    """Set Type of Calculation."""

    Bishop = 1
    UpliftVan = 2
    Spencer = 3


class DStabilityObject(BaseModel, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _to_dstability_sub_structure(self):
        raise NotImplementedError


class DStabilityModel(BaseModel):
    """D-Stability is software for soft soil slope stability.

    This model can read, modify and create
    .stix files
    """

    current_scenario: int = -1
    current_stage: int = -1
    current_calculation: int = -1
    datastructure: DStabilityStructure = DStabilityStructure()
    current_id: int = -1

    def __init__(self, *args, **data) -> None:
        super().__init__(*args, **data)
        self.current_id = self.datastructure.get_unique_id()

    @property
    def parser_provider_type(self) -> Type[DStabilityParserProvider]:
        return DStabilityParserProvider

    @property
    def default_console_path(self) -> Path:
        return Path("DStabilityConsole/D-Stability Console.exe")

    @property
    def custom_console_path(self) -> Path:
        return self.get_meta_property("dstability_console_path")

    @property
    def soils(self) -> SoilCollection:
        """Enables easy access to the soil in the internal dict-like datastructure. Also enables edit/delete for individual soils."""
        return self.datastructure.soils

    @property
    def soil_correlations(self) -> SoilCorrelation:
        return self.datastructure.soilcorrelation

    @property
    def zmax(self) -> float:
        geometry = self._get_geometry(self.current_scenario, self.current_stage)
        return geometry.zmax

    @property
    def zmin(self) -> float:
        geometry = self._get_geometry(self.current_scenario, self.current_stage)
        return geometry.zmin

    @property
    def xmax(self) -> float:
        geometry = self._get_geometry(self.current_scenario, self.current_stage)
        return geometry.xmax

    @property
    def xmin(self) -> float:
        geometry = self._get_geometry(self.current_scenario, self.current_stage)
        return geometry.xmin

    @property
    def surface(self) -> List[Tuple[float, float]]:
        """Get the surface of the model as a list of x,z tuples

        Returns:
            List[Tuple[float, float]]: The top surface of the model
        """
        geometry = self._get_geometry(self.current_scenario, self.current_stage)
        return geometry.surface

    @property
    def ditch_points(self) -> List[Tuple[float, float]]:
        """Get the ditch points from left (riverside) to right (landside), this will return
        the ditch embankement side, ditch embankement side bottom, land side bottom, land side
        or empty list if there are not ditch points

        Returns:
            List[Tuple[float, float]]: List of points or empty list if no ditch is found
        """

        p1 = self.get_characteristic_point(CharacteristicPointEnum.DITCH_EMBANKEMENT_SIDE)
        p2 = self.get_characteristic_point(
            CharacteristicPointEnum.DITCH_BOTTOM_EMBANKEMENT_SIDE
        )
        p3 = self.get_characteristic_point(CharacteristicPointEnum.DITCH_BOTTOM_LAND_SIDE)
        p4 = self.get_characteristic_point(CharacteristicPointEnum.DITCH_LAND_SIDE)

        if p1 and p2 and p3 and p4:
            return [
                (p1.x, self.z_at(p1.x)),
                (p2.x, self.z_at(p2.x)),
                (p3.x, self.z_at(p3.x)),
                (p4.x, self.z_at(p4.x)),
            ]
        else:
            return []

    @property
    def waternets(self) -> List[Waternet]:
        return self.datastructure.waternets

    @property
    def output(self) -> List[DStabilityResult]:
        def _get_result_or_none(scenario_index, calculation_index) -> DStabilityResult:
            if self.has_result(
                scenario_index=int(scenario_index),
                calculation_index=int(calculation_index),
            ):
                return self.get_result(
                    scenario_index=int(scenario_index),
                    calculation_index=int(calculation_index),
                )
            else:
                return None

        all_results = []

        for scenario_index, scenario in enumerate(self.datastructure.scenarios):
            for calculation_index, _ in enumerate(scenario.Calculations):
                all_results.append(
                    _get_result_or_none(
                        scenario_index=scenario_index, calculation_index=calculation_index
                    )
                )

        return all_results

    @property
    def layer_soil_dict(self) -> Dict:
        """Get the soils as a dictionary of the layer id

        Returns:
            Dict: A dictionary containing the layer id's as key and the PersistibleSoil as value
        """
        result = {}
        for layer in self._get_geometry(self.current_scenario, self.current_stage).Layers:
            for soillayer in self._get_soil_layers(
                self.current_scenario, self.current_stage
            ).SoilLayers:
                if layer.Id == soillayer.LayerId:
                    for soil in self.soils.Soils:
                        if soil.Id == soillayer.SoilId:
                            result[layer.Id] = soil
        return result

    @property
    def phreatic_line(self) -> Optional[PersistableHeadLine]:
        wnet = self._get_waternet(self.current_scenario, self.current_stage)
        phreatic_headline_id = wnet.PhreaticLineId

        if phreatic_headline_id is None:
            return None

        for headline in wnet.HeadLines:
            if headline.Id == phreatic_headline_id:
                return headline

    def _get_next_id(self) -> int:
        self.current_id += 1
        return self.current_id

    def parse(self, *args, **kwargs):
        try:
            super().parse(*args, **kwargs)
        except ValueError as e:
            if e.args[0] == "Can't listdir a file":
                executable = meta.dstability_migration_console_path
                if not executable.exists():
                    logger.error(
                        f"The path to the dstability migration console (geolib.env) is not set or invalid`{executable}`, cannot auto convert this file"
                    )
                    raise CalculationError(
                        -1,
                        f"DStability Migration Console executable not set or not found at {executable}.",
                    )
                try:
                    subprocess.run([executable, self.filename, self.filename])
                    self.parse(self.filename)
                except Exception as e:
                    logger.error(
                        f"Error running the migration console on this file; '{e}'"
                    )
                    raise CalculationError(f"Cannot open stix file, got error '{e}'")

        self.current_id = self.datastructure.get_unique_id()

    def layer_at(self, x: float, z: float) -> Optional[PersistableLayer]:
        """Get the layer at the given x, z coordinate

        Args:
            x (float): The x coordinate
            z (float): The z coordinate

        Returns:
            Optional[PersistableLayer]: The layer of None if no layer is found
        """
        geometry = self._get_geometry(self.current_scenario, self.current_stage)
        return geometry.layer_at(x, z)

    def z_at(
        self, x: float, highest_only: bool = True
    ) -> Optional[Union[float, List[float]]]:
        """Get a list of z coordinates from intersections with the soillayers on coordinate x

        Args:
            x (_type_): The x coordinate
            highest_only (bool): Only return the topmost point. Defaults to True

        Returns:
            List[float]: A list of intersections sorted from high to low or only the highest point if highest_only is True
        """
        intersections = self.layer_intersections_at(x)

        if len(intersections) > 0:
            if highest_only:
                return intersections[0][0]
            else:
                return sorted(
                    [i[0] for i in intersections] + [intersections[-1][1]], reverse=True
                )
        else:
            return None

    def phreatic_level_at(self, x: float) -> Optional[float]:
        phreatic_line = self.phreatic_line
        if phreatic_line is None:
            return None

        phreatic_line_points = [(p.X, p.Z) for p in phreatic_line.Points]
        intersections = polyline_polyline_intersections(
            [(x, self.zmax + 0.01), (x, self.zmin - 0.01)], phreatic_line_points
        )

        if len(intersections) == 0:
            return None

        return intersections[0][1]

    def stresses_at(
        self, x: float, include_loads: bool = False
    ) -> List[Tuple[float, float, float, float]]:
        result = []
        if include_loads:
            raise NotImplementedError(
                "Including loads in the stresses calculation is not added yet"
            )
        layers = self.layer_intersections_at(x)

        if len(layers) == 0:
            return result

        phreatic_level = self.phreatic_level_at(x)

        if phreatic_level is None:
            phreatic_level = layers[-1][1] - 0.01

        stot, u = 0.0, 0.0
        if layers[0][0] < phreatic_level:
            result.append((phreatic_level, 0.0, 0.0, 0.0))
            u += (phreatic_level - layers[0][0]) * UNIT_WEIGHT_WATER
            stot = u
            result.append((layers[0][0], stot, u, 0.0))
        else:
            result.append((layers[0][0], stot, u, 0.0))

        for layer in layers:
            if layer[0] <= phreatic_level:
                stot += layer[2].VolumetricWeightBelowPhreaticLevel * (
                    layer[0] - layer[1]
                )
                u = max((phreatic_level - layer[1]) * UNIT_WEIGHT_WATER, 0.0)
                result.append((layer[1], stot, u, max(stot - u, 0)))
            elif layer[1] >= phreatic_level:
                stot += layer[2].VolumetricWeightAbovePhreaticLevel * (
                    layer[0] - layer[1]
                )
                u = max((phreatic_level - layer[1]) * UNIT_WEIGHT_WATER, 0.0)
                result.append((layer[1], stot, u, max(stot - u, 0)))
            else:
                stot += layer[2].VolumetricWeightAbovePhreaticLevel * (
                    layer[0] - phreatic_level
                )
                result.append((layer[1], stot, 0.0, max(stot - u, 0)))
                stot += layer[2].VolumetricWeightAbovePhreaticLevel * (
                    phreatic_level - layer[1]
                )
                u = max((phreatic_level - layer[1]) * UNIT_WEIGHT_WATER, 0.0)
                result.append((layer[1], stot, u, max(stot - u, 0)))

        return result

    def layer_by_id(self, layer_id: str) -> Optional[PersistableLayer]:
        geometry = self._get_geometry(self.current_scenario, self.current_stage)
        for layer in geometry.Layers:
            if layer.Id == layer_id:
                return layer
        return None

    def layer_by_label(self, layer_label: str) -> Optional[PersistableLayer]:
        geometry = self._get_geometry(self.current_scenario, self.current_stage)
        for layer in geometry.Layers:
            if layer.Label == layer_label:
                return layer
        return None

    def get_characteristic_point(
        self, characteristic_point_type: CharacteristicPointEnum
    ) -> Optional[Point]:
        """Get the point value for the given characteristic point type

        Args:
            characteristic_point_type (CharacteristicPointEnum): Type of characteristic point

        Returns:
            Optional[Point]: The point or None if not found
        """

        def convert_wncs_point(x: Union[float, str]) -> Optional[Point]:
            if x != "NaN":
                x = float(x)
                return x
            else:
                return None

        wncs = self._get_waternetcreator_settings(
            self.current_scenario, self.current_stage
        )
        if wncs is None:
            return None

        x = None

        if characteristic_point_type == CharacteristicPointEnum.NONE:
            return None
        elif (
            characteristic_point_type
            == CharacteristicPointEnum.EMBANKEMENT_TOE_WATER_SIDE
        ):
            x = convert_wncs_point(wncs.EmbankmentCharacteristics.EmbankmentToeWaterSide)
        elif (
            characteristic_point_type
            == CharacteristicPointEnum.EMBANKEMENT_TOP_WATER_SIDE
        ):
            x = convert_wncs_point(wncs.EmbankmentCharacteristics.EmbankmentTopWaterSide)
        elif (
            characteristic_point_type == CharacteristicPointEnum.EMBANKEMENT_TOP_LAND_SIDE
        ):
            x = convert_wncs_point(wncs.EmbankmentCharacteristics.EmbankmentTopLandSide)
        elif characteristic_point_type == CharacteristicPointEnum.SHOULDER_BASE_LAND_SIDE:
            x = convert_wncs_point(wncs.EmbankmentCharacteristics.ShoulderBaseLandSide)
        elif (
            characteristic_point_type == CharacteristicPointEnum.EMBANKEMENT_TOE_LAND_SIDE
        ):
            x = convert_wncs_point(wncs.EmbankmentCharacteristics.EmbankmentToeLandSide)

        elif characteristic_point_type == CharacteristicPointEnum.DITCH_EMBANKEMENT_SIDE:
            x = convert_wncs_point(wncs.DitchCharacteristics.DitchEmbankmentSide)
        elif (
            characteristic_point_type
            == CharacteristicPointEnum.DITCH_BOTTOM_EMBANKEMENT_SIDE
        ):
            x = convert_wncs_point(wncs.DitchCharacteristics.DitchBottomEmbankmentSide)
        elif characteristic_point_type == CharacteristicPointEnum.DITCH_BOTTOM_LAND_SIDE:
            x = convert_wncs_point(wncs.DitchCharacteristics.DitchBottomLandSide)
        elif characteristic_point_type == CharacteristicPointEnum.DITCH_LAND_SIDE:
            x = convert_wncs_point(wncs.DitchCharacteristics.DitchLandSide)
        else:
            raise NotImplementedError(
                f"No code to deal with point type '{characteristic_point_type}' yet!"
            )

        if isnan(x):
            return None
        else:
            return Point(x=x, z=self.z_at(x))

    def generate_waternet(
        self,
        river_level_mhw: float,
        river_level_ghw: float,
        polder_level: float,
        B_offset: Optional[float] = None,
        C_offset: Optional[float] = None,
        E_offset: Optional[float] = None,
        D_offset: Optional[float] = None,
        surface_offset: float = 0.01,
        x_embankment_toe_land_side: Optional[
            float
        ] = None,  # if None we try to find it using the waternetcreator settings
        x_embankment_top_land_side: Optional[
            float
        ] = None,  # if None we try to find it using the waternetcreator settings
        x_shoulder_base_land_side: Optional[
            float
        ] = None,  # if None we try to find it using the waternetcreator settings
        x_embankment_toe_water_side: Optional[
            float
        ] = None,  # if None we try to find it using the waternetcreator settings
        material_layout: Optional[EmbankmentSoilScenarioEnum] = None,
        phreatic_level_embankment_top_waterside: Optional[float] = None,
        phreatic_level_embankment_top_landside: Optional[float] = None,
        aquifer_label: Optional[str] = None,  # it is possible to use the label or the id
        aquifer_id: Optional[str] = (
            None  # NOTE if the id is set, this will be used BEFORE the label
        ),
        aquifer_inside_aquitard_label: Optional[str] = (
            None  # it is possible to use the label or the id
        ),
        aquifer_inside_aquitard_id: Optional[str] = (
            None  # NOTE if the id is set, this will be used BEFORE the label
        ),
        intrusion_length: Optional[float] = 1.0,  # default 1.0 or 3.0 for tidal zones
        hydraulic_head_pl2_inward: Optional[float] = None,
        hydraulic_head_pl2_outward: Optional[float] = None,
        inward_leakage_length_pl3: Optional[float] = None,
        outward_leakage_length_pl3: Optional[float] = None,
        inward_leakage_length_pl4: Optional[float] = None,
        outward_leakage_length_pl4: Optional[float] = None,
        adjust_for_uplift: Optional[bool] = None,
    ):
        def f_phi2(x, phi2_in, phi2_out, x_left, x_right) -> float:
            return phi2_out + (phi2_in - phi2_out) / (x_right - x_left) * (x - x_left)

        def f_phi34(mhw, ghw, gamma_pl_out, gamma_pl_in, phi2) -> float:
            return phi2 + (mhw - ghw) / (1 + gamma_pl_out / gamma_pl_in)

        def f_phi3_x(phi3_crest_out, phi2_crest_out, dX, lambda_pl3_in, phi2_x) -> float:
            return (phi3_crest_out - phi2_crest_out) * exp(-dX / lambda_pl3_in) + phi2_x

        def f_phi4_x(phi3_crest_out, phi2_crest_out, dX, lambda_pl4_in, phi2_x) -> float:
            return (phi3_crest_out - phi2_crest_out) * exp(-dX / lambda_pl4_in) + phi2_x

        wncs = self._get_waternetcreator_settings(
            self.current_scenario, self.current_stage
        )
        if x_embankment_toe_land_side is None:
            pt_embankment_toe_land_side = self.get_characteristic_point(
                CharacteristicPointEnum.EMBANKEMENT_TOE_LAND_SIDE
            )
            if pt_embankment_toe_land_side is None:
                raise CalculationError(
                    f"Cannot generate the phreatic line because the x coordinate for the embankment toe land side is not given."
                )
        else:
            pt_embankment_toe_land_side = Point(
                x=x_embankment_toe_land_side, z=self.z_at(x_embankment_toe_land_side)
            )

        if x_embankment_top_land_side is None:
            pt_embankment_top_land_side = self.get_characteristic_point(
                CharacteristicPointEnum.EMBANKEMENT_TOP_LAND_SIDE
            )
            if pt_embankment_top_land_side is None:
                raise CalculationError(
                    f"Cannot generate the phreatic line because the x coordinate for the embankment top land side is not given."
                )

        if x_shoulder_base_land_side is None:
            pt_shoulder_base_land_side = self.get_characteristic_point(
                CharacteristicPointEnum.SHOULDER_BASE_LAND_SIDE
            )
        else:
            pt_shoulder_base_land_side = None

        # get setting from the waternet creator settings if they are None
        if B_offset is None:
            B_offset = wncs.OffsetEmbankmentTopWaterSide
            if isnan(B_offset):
                B_offset = None
        if C_offset is None:
            C_offset = wncs.OffsetEmbankmentTopLandSide
            if isnan(C_offset):
                C_offset = None
        if D_offset is None:
            D_offset = wncs.OffsetShoulderBaseLandSide
            if isnan(D_offset):
                D_offset = None
        if E_offset is None:
            E_offset = wncs.OffsetEmbankmentToeLandSide
            if isnan(E_offset):
                E_offset = None
        if material_layout is None:
            material_layout = wncs.EmbankmentSoilScenario
        if phreatic_level_embankment_top_waterside is None:
            phreatic_level_embankment_top_waterside = (
                wncs.InitialLevelEmbankmentTopWaterSide
            )
        if phreatic_level_embankment_top_landside is None:
            phreatic_level_embankment_top_landside = (
                wncs.InitialLevelEmbankmentTopLandSide
            )
        if aquifer_label is None and aquifer_id is None:
            aquifer_id = wncs.AquiferLayerId
        if aquifer_inside_aquitard_label is None and aquifer_inside_aquitard_id is None:
            aquifer_inside_aquitard_id = wncs.AquiferInsideAquitardLayerId
        if intrusion_length is None:
            intrusion_length = wncs.IntrusionLength
        if hydraulic_head_pl2_inward is None:
            hydraulic_head_pl2_inward = wncs.AquitardHeadWaterSide
        if hydraulic_head_pl2_outward is None:
            hydraulic_head_pl2_outward = wncs.AquitardHeadLandSide
        if inward_leakage_length_pl3 is None:
            inward_leakage_length_pl3 = wncs.PleistoceneLeakageLengthInwards
        if outward_leakage_length_pl3 is None:
            outward_leakage_length_pl3 = wncs.PleistoceneLeakageLengthOutwards
        if inward_leakage_length_pl4 is None:
            inward_leakage_length_pl4 = (
                wncs.AquiferLayerInsideAquitardLeakageLengthInwards
            )
        if outward_leakage_length_pl4 is None:
            outward_leakage_length_pl4 = (
                wncs.AquiferLayerInsideAquitardLeakageLengthOutwards
            )
        if adjust_for_uplift is None:
            adjust_for_uplift = wncs.AdjustForUplift

        # if we have an aquifer, get the layer and check the input
        aquifer_layer = None
        if aquifer_id is not None or aquifer_label is not None:
            # get the layer
            if aquifer_id is not None:
                aquifer_layer = self.layer_by_id(aquifer_id)
            elif aquifer_label is not None:
                aquifer_layer = self.layer_by_label(aquifer_label)

            # did we manage to get the layer?
            if aquifer_layer is None:
                raise WaternetCreatorError(
                    "Could not get the aquifer layer by the given name or label"
                )
            # and if we have a layer we also need the next parameters
            if inward_leakage_length_pl3 is None or outward_leakage_length_pl3 is None:
                raise WaternetCreatorError(
                    "PL3 is set but the inward and/or outward leakage lengths are missing."
                )

        # if we have an aquifer, get the layer and check the input
        aquifer_inside_aquitard_layer = None
        if (
            aquifer_inside_aquitard_id is not None
            or aquifer_inside_aquitard_label is not None
        ):
            # get the layer
            if aquifer_inside_aquitard_id is not None:
                aquifer_inside_aquitard_layer = self.layer_by_id(
                    aquifer_inside_aquitard_id
                )
            elif aquifer_label is not None:
                aquifer_inside_aquitard_layer = self.layer_by_label(
                    aquifer_inside_aquitard_label
                )

            # did we manage to get the layer?
            if aquifer_inside_aquitard_layer is None:
                raise WaternetCreatorError(
                    "Could not get the aquifer inside aquitard layer by the given name or label"
                )
            # and if we have a layer we also need the next parameters
            if inward_leakage_length_pl4 is None or outward_leakage_length_pl4 is None:
                raise WaternetCreatorError(
                    f"PL4 is set but the inward and/or outward leakage lengths are missing."
                )

        h = river_level_mhw - self.z_at(pt_embankment_toe_land_side.x)

        # point A - (ALL) - Intersection of the river water level with the outer slope
        intersections = polyline_polyline_intersections(
            [(self.xmin, river_level_mhw), (self.xmax, river_level_mhw)], self.surface
        )

        if len(intersections) == 0:
            raise WaternetCreatorError(
                f"No intersection with the surface and the given river level ({self.river_level_mhw}) found."
            )

        Ax, Az = intersections[0]

        # Point E is needed for interpolation so we calculate this first
        # Point E1 (CLAY DIKE) Surface level at dike toe minus offset, with default offset 0 m
        Ex = pt_embankment_toe_land_side.x
        Ez1 = self.z_at(Ex)
        if E_offset is not None:
            # user defined offset (if no user defined offset the the default offset equals zero)
            Ez1 -= E_offset

        # Point E2 (SAND ON CLAY) Surface level at dike toe minus offset, with default offset −0.25 × (river level - polder level).
        Ez2 = self.z_at(Ex)
        if E_offset is None:
            Ez2 += 0.25 * h
        else:
            Ez2 += E_offset

        # Point E3 (SAND ON CLAY) = E2 (SAND ON SAND)
        Ez3 = Ez2

        # Point E must be equal to or above polder level
        Ez1 = max(polder_level, Ez1)
        Ez2 = max(polder_level, Ez2)
        Ez3 = max(polder_level, Ez3)

        # Point B1 (CLAY DIKE) River water level minus offset, with default offset 1 m, limited by minimum value ZB;initial, see section 3.3.1.2.
        # TODO
        # There seems to be an inconsitency in DStability
        # If you fill in 0.0 for the values under Phreatic level in embankment at points
        # Then the z values will be calculated but if you fill in another value
        # these will be used as the z values
        # in other words, it is not possible to fill in 0.0 as the actual waterlevel
        # Unfortunately we need to also make use of this method
        Bx1 = Ax + 1.0
        if (
            phreatic_level_embankment_top_waterside is not None
            and phreatic_level_embankment_top_waterside != 0.0
        ):
            # user defined pl
            Bz1 = phreatic_level_embankment_top_waterside
        elif B_offset is not None:
            # user defined offset
            Bz1 = river_level_mhw - B_offset
        else:  # default offset
            Bz1 = river_level_mhw - 1.0

        # Point B2 (SAND ON CLAY) River water level minus offset, with default offset 0.5 × (river level - dike toe polder level), limited by minimum value ZB;initial, see section 3.3.1.2.
        Bx2 = Ax + 0.001
        if (
            phreatic_level_embankment_top_waterside is not None
            and phreatic_level_embankment_top_waterside != 0.0
        ):
            # user defined pl
            Bz2 = phreatic_level_embankment_top_waterside
        elif B_offset is not None:
            # user defined offset
            Bz2 = river_level_mhw - B_offset
        else:
            # default offset
            Bz2 = river_level_mhw - 0.5 * h

        # Point B3 (SAND ON SAND) Linear interpolation between point A and point E, limited by minimum value ZB;initial, see section 3.3.1.2.
        Bx3 = Ax + 0.001
        if (
            phreatic_level_embankment_top_waterside is not None
            and phreatic_level_embankment_top_waterside != 0.0
        ):
            # user defined pl
            Bz3 = phreatic_level_embankment_top_waterside
        else:
            # lineair interpolation
            Bz3 = Az + (Bx3 - Ax) / (Ex - Ax) * (Ez3 - Az)

        # Point C1 (CLAY DIKE) River water level minus offset, with default offset 1.5 m, limited by minimum value ZC;initial, see section 3.3.1.2.
        # TODO > Same inconsistency as point B, only solvable if DStab is fixed
        Cx = pt_embankment_top_land_side.x
        if (
            phreatic_level_embankment_top_landside is not None
            and phreatic_level_embankment_top_landside != 0.0
        ):
            Cz1 = phreatic_level_embankment_top_landside
        elif C_offset is not None:
            Cz1 = river_level_mhw - C_offset
        else:
            Cz1 = river_level_mhw - 1.5

        # Point C2 (SAND ON CLAY) Linear interpolation between point B and point E, limited by minimum value ZC;initial, see section 3.3.1.2.
        if (
            phreatic_level_embankment_top_landside is not None
            and phreatic_level_embankment_top_landside != 0.0
        ):
            Cz2 = phreatic_level_embankment_top_landside
        else:
            Cz2 = Bz2 + (Cx - Bx2) / (Ex - Bx2) * (Ez2 - Bz2)

        # Point C3 (SAND ON SAND) Linear interpolation between point A and point E, limited by minimum value ZC;initial, see section 3.3.1.2.
        if (
            phreatic_level_embankment_top_landside is not None
            and phreatic_level_embankment_top_landside != 0.0
        ):
            Cz3 = phreatic_level_embankment_top_landside
        else:
            Cz3 = Az + (Cx - Ax) / (Ex - Ax) * (Ez3 - Az)

        if pt_shoulder_base_land_side is not None:
            Dx = pt_shoulder_base_land_side.x

            # Point D1 (CLAY DIKE) Linear interpolation between point C and point E, unless the user defines an offset Doffset;user with respect to the surface level.
            if D_offset is not None:
                Dz1 = self.z_at(Dx) - D_offset
            else:
                Dz1 = Cz1 + (Dx - Cx) / (Ex - Cx) * (Ez1 - Cz1)

            # Point D2 (SAND ON CLAY) Linear interpolation between point B and point E, unless the user defines an offset Doffset;user with respect to the surface level.
            if D_offset is not None:
                Dz2 = self.z_at(Dx) - D_offset
            else:
                Dz2 = Bz2 + (Dx - Bx2) / (Ex - Bx2) * (Ez2 - Bz2)

            # Point D3 (SAND ON SAND) Linear interpolation between point A and point E, unless the user defines an offset Doffset;user with respect to the surface level
            if D_offset is not None:
                Dz3 = self.z_at(Dx) - D_offset
            else:
                Dz3 = Az + (Dx - Ax) / (Ex - Ax) * (Ez3 - Az)
        else:  # add D as lin interpolated point between C and E
            Dx = (Ex + Cx) / 2.0
            Dz1 = (Ez1 + Cz1) / 2.0
            Dz2 = (Ez2 + Cz2) / 2.0
            Dz3 = (Ez3 + Cz3) / 2.0

        # Point D must be equal to or above polder level
        Dz1 = max(polder_level, Dz1)
        Dz2 = max(polder_level, Dz2)
        Dz3 = max(polder_level, Dz3)

        # Point D must be equal to or below point C
        Dz1 = min(Cz1, Dz1)
        Dz2 = min(Cz2, Dz2)
        Dz3 = min(Cz3, Dz3)

        # Point E must be equal to or below point D
        # TODO > what if we adjust this value but E is already used for interpolation
        Ez1 = min(Dz1, Ez1)
        Ez2 = min(Dz2, Ez2)
        Ez3 = min(Dz3, Ez3)

        # Point F Intersection point polder level with ditch (is determined automatically)
        Fz = polder_level
        if len(self.ditch_points) > 0:  # with ditch find intersection
            intersections = polyline_polyline_intersections(
                [(self.xmin, polder_level), (self.xmax, polder_level)], self.ditch_points
            )
            if intersections is not None and len(intersections) > 0:
                Fx1, _ = intersections[0]
                Fx2 = Fx1
                Fx3 = Fx1
        else:
            # If no ditch, lin extrapolation to polder level from C to E
            Fx1 = Ex + (Ez1 - Fz) * (Ex - Cx) / (Ez1 - Cz1)
            Fx2 = Ex + (Ez2 - Fz) * (Ex - Cx) / (Ez2 - Cz2)
            Fx3 = Ex + (Ez3 - Fz) * (Ex - Cx) / (Ez3 - Cz3)

        # TODO is it possible that wncs.EmbankmentSoilScenario can be None?
        if (
            material_layout == EmbankmentSoilScenarioEnum["CLAY_EMBANKMENT_ON_CLAY"]
            or material_layout == EmbankmentSoilScenarioEnum["CLAY_EMBANKMENT_ON_SAND"]
        ):
            abcdef = [[Ax, Az], [Bx1, Bz1], [Cx, Cz1], [Dx, Dz1], [Ex, Ez1], [Fx1, Fz]]
        elif material_layout == EmbankmentSoilScenarioEnum["SAND_EMBANKMENT_ON_CLAY"]:
            abcdef = [[Ax, Az], [Bx2, Bz2], [Cx, Cz2], [Dx, Dz2], [Ex, Ez2], [Fx2, Fz]]
        elif material_layout == EmbankmentSoilScenarioEnum["SAND_EMBANKMENT_ON_SAND"]:
            abcdef = [[Ax, Az], [Bx3, Bz3], [Cx, Cz3], [Dx, Dz3], [Ex, Ez3], [Fx3, Fz]]
        else:
            raise ValueError(f"Unknown EmbankmentSoilScenarioEnum '{material_layout}'")

        # Make sure the phreatic line does not exceed the surface
        # Between A and F
        # 1. get all surface points
        surface_points = [
            p for p in self.surface if p[0] > abcdef[0][0] and p[0] < abcdef[-1][0]
        ]
        # 2. get all intersections between surface and pl line
        intersections = [
            p
            for p in polyline_polyline_intersections(abcdef, self.surface)
            if p[0] > Ax and p[0] < Ex
        ]
        # 3. merge x coords
        check_points = sorted(
            list(set([p[0] for p in surface_points + intersections + abcdef[1:-1]]))
        )

        # create the final points, start with the leftmost point
        final_points = [[self.xmin, river_level_mhw], abcdef[0]]
        # now add the points
        for x in check_points:
            for i in range(1, len(abcdef)):
                x1, z1 = abcdef[i - 1]
                x2, z2 = abcdef[i]
                if x1 <= x and x <= x2:
                    z_pl = z1 + (x - x1) / (x2 - x1) * (z2 - z1)
                    z_surface = self.z_at(x)

                    if z_pl > z_surface - surface_offset:
                        z_pl = z_surface - surface_offset

                    if z_pl > final_points[-1][1]:
                        z_pl = final_points[-1][1]

                    final_points.append([x, z_pl])
                    break

        # add F and the rightmost point
        final_points += [abcdef[-1], [self.xmax, polder_level]]

        # clear current headlines and referencelines
        wnet = self._get_waternet(self.current_scenario, self.current_stage)
        wnet.HeadLines.clear()
        wnet.ReferenceLines.clear()

        # Add phreatic line
        pl_id = self.add_head_line(
            [Point(x=p[0], z=p[1]) for p in final_points],
            "Stijghoogtelijn (PL 1)",
            is_phreatic_line=True,
            scenario_index=self.current_scenario,
            stage_index=self.current_stage,
        )

        # Add the referenceline for the phreatic zone top
        self.add_reference_line(
            points=[Point(x=p[0], z=p[1]) for p in self.surface],
            top_head_line_id=pl_id,
            bottom_headline_id=pl_id,
            scenario_index=self.current_scenario,
            stage_index=self.current_stage,
            label="Freatische zone bovenkant",
        )

        # Add the referenceline for the phreatic zone bottom 1 (bottom of the top layer)
        geometry = self._get_geometry(self.current_scenario, self.current_stage)
        toplayer_points = [(p.X, p.Z) for p in geometry.top_layer().Points]
        bottom_line_pl1 = bottom_of_polygon(toplayer_points)
        self.add_reference_line(
            points=[Point(x=p[0], z=p[1]) for p in bottom_line_pl1],
            top_head_line_id=pl_id,
            bottom_headline_id=pl_id,
            scenario_index=self.current_scenario,
            stage_index=self.current_stage,
            label="Freatische zone onderkant (1)",
        )

        # For scenario “Sand dike on sand” (2B), only PL1 (Phreatic line) is created
        if material_layout == EmbankmentSoilScenarioEnum["SAND_EMBANKMENT_ON_SAND"]:
            return

        # if we have no aquifer then we have no PL2, PL3 or PL4
        if aquifer_label is None and aquifer_id is None:
            return

        aquifer_points = top_of_polygon(
            [[float(p.X), float(p.Z)] for p in aquifer_layer.Points]
        )
        i = 1

        #######
        # PL2 #
        #######
        # if some parameters are not set, get them from the waternet creator settings

        # if not set use the defaults
        if intrusion_length is not None:
            if hydraulic_head_pl2_inward is None:
                hydraulic_head_pl2_inward = river_level_ghw
            else:
                hydraulic_head_pl2_inward = min(
                    hydraulic_head_pl2_inward, river_level_ghw
                )
            if hydraulic_head_pl2_outward is None:
                hydraulic_head_pl2_outward = river_level_ghw
            else:
                hydraulic_head_pl2_outward = min(
                    hydraulic_head_pl2_outward, river_level_ghw
                )

            pl2points = [
                [self.xmin, hydraulic_head_pl2_inward],
                [self.xmax, hydraulic_head_pl2_outward],
            ]

            # add the headline
            pl2_id = self.add_head_line(
                points=[Point(x=p[0], z=p[1]) for p in pl2points],
                label="Stijghoogtelijn 2 (PL2)",
                scenario_index=self.current_scenario,
                stage_index=self.current_stage,
            )

            # add the penetration_layer_thickness
            aquifer_points_with_penetration_layer = [
                [p[0], p[1] + intrusion_length] for p in aquifer_points
            ]

            # add the referenceline
            self.add_reference_line(
                points=[
                    Point(x=p[0], z=p[1]) for p in aquifer_points_with_penetration_layer
                ],
                bottom_headline_id=pl2_id,
                top_head_line_id=pl2_id,
                label="Indringingszone onderste aquifer",
                scenario_index=self.current_scenario,
                stage_index=self.current_stage,
            )

        #######
        # PL3 #
        #######
        if x_embankment_toe_water_side is None:
            pt_embankment_toe_water_side = self.get_characteristic_point(
                CharacteristicPointEnum.EMBANKEMENT_TOE_WATER_SIDE
            )
            if pt_embankment_top_land_side is None:
                raise CalculationError(
                    f"Cannot generate PL3 because the x coordinate for the embankment toe water side is not given."
                )

        A1B = [self.xmin, river_level_mhw]
        B1B = [pt_embankment_toe_water_side.x, river_level_mhw]

        # SCENARIO 1B - Clay on sand
        if material_layout == EmbankmentSoilScenarioEnum["CLAY_EMBANKMENT_ON_SAND"]:
            D1B = (Ex, self.z_at(Ex))
            G1B = (self.xmax, D1B[1])

            pl3_id = self.add_head_line(
                points=[Point(x=p[0], z=p[1]) for p in [A1B, B1B, D1B, G1B]],
                label="Stijghoogtelijn 3 (PL3)",
                scenario_index=self.current_scenario,
                stage_index=self.current_stage,
            )

            self.add_reference_line(
                points=[Point(x=p[0], z=p[1]) for p in aquifer_points],
                bottom_headline_id=pl3_id,
                top_head_line_id=pl3_id,
                label="Waternetlijn onderste aquifer",
                scenario_index=self.current_scenario,
                stage_index=self.current_stage,
            )
        else:  # SCENARIO 1A and 2A - sand on clay / clay on clay
            x_embankment_top_water_side = pt_embankment_top_land_side.x
            phi2_embankment_top_waterside = f_phi2(
                x_embankment_top_water_side,
                hydraulic_head_pl2_inward,
                hydraulic_head_pl2_outward,
                self.xmin,
                self.xmax,
            )

            C1A_PL3 = [
                x_embankment_top_water_side,
                f_phi34(
                    river_level_mhw,
                    river_level_ghw,
                    outward_leakage_length_pl3,
                    inward_leakage_length_pl3,
                    phi2_embankment_top_waterside,
                ),
            ]
            if aquifer_inside_aquitard_id is not None:
                C1A_PL4 = [
                    x_embankment_top_water_side,
                    f_phi34(
                        river_level_mhw,
                        river_level_ghw,
                        outward_leakage_length_pl4,
                        inward_leakage_length_pl4,
                        phi2_embankment_top_waterside,
                    ),
                ]

            # NOTE phi3_crest_out = C1A_PL3[1]
            #      phi4_crest_out = C1A_PL4[1]
            if not adjust_for_uplift:
                x_embankment_toe_land_side = pt_embankment_toe_land_side.x
                phi2_crest_out = f_phi2(
                    x_embankment_top_water_side,
                    hydraulic_head_pl2_inward,
                    hydraulic_head_pl2_outward,
                    self.xmin,
                    self.xmax,
                )
                phi_2_embankment_toe_land_side = f_phi2(
                    x_embankment_toe_land_side,
                    hydraulic_head_pl2_inward,
                    hydraulic_head_pl2_outward,
                    self.xmin,
                    self.xmax,
                )
                dx = x_embankment_toe_land_side - x_embankment_top_water_side
                z_d1a = phi_2_embankment_toe_land_side + (
                    C1A_PL3[1] - phi2_crest_out
                ) * exp(-dx / inward_leakage_length_pl3)
                D1A = [x_embankment_toe_land_side, z_d1a]

                if aquifer_inside_aquitard_layer is not None:
                    z_d2a = phi_2_embankment_toe_land_side + (
                        C1A_PL3[1] - phi2_crest_out
                    ) * exp(-dx / inward_leakage_length_pl4)
                    D2A = [x_embankment_toe_land_side, z_d2a]

                dx = self.xmax - x_embankment_top_water_side
                z_g1a = phi_2_embankment_toe_land_side + (
                    C1A_PL3[1] - phi2_crest_out
                ) * exp(-dx / inward_leakage_length_pl3)
                G1A = [self.xmax, z_g1a]

                if aquifer_inside_aquitard_layer is not None:
                    z_g2a = phi_2_embankment_toe_land_side + (
                        C1A_PL3[1] - phi2_crest_out
                    ) * exp(-dx / inward_leakage_length_pl4)
                    G2A = [self.xmax, z_g2a]

                pl3_id = self.add_head_line(
                    points=[Point(x=p[0], z=p[1]) for p in [A1B, B1B, C1A_PL3, D1A, G1A]],
                    label="Stijghoogtelijn 3 (PL3)",
                    scenario_index=self.current_scenario,
                    stage_index=self.current_stage,
                )

                self.add_reference_line(
                    points=[Point(x=p[0], z=p[1]) for p in aquifer_points],
                    bottom_headline_id=pl3_id,
                    top_head_line_id=pl3_id,
                    label="Waternetlijn onderste aquifer",
                    scenario_index=self.current_scenario,
                    stage_index=self.current_stage,
                )
                if aquifer_inside_aquitard_layer is not None:
                    pl4_id = self.add_head_line(
                        points=[
                            Point(x=p[0], z=p[1]) for p in [A1B, B1B, C1A_PL4, D2A, G2A]
                        ],
                        label="Stijghoogtelijn 4 (PL4)",
                        scenario_index=self.current_scenario,
                        stage_index=self.current_stage,
                    )
                    # add referenceline but where...

            else:
                raise WaternetCreatorError(
                    "Scenario with correction for uplift not yet implemented!"
                )

    def has_result(
        self,
        scenario_index: Optional[int] = None,
        calculation_index: Optional[int] = None,
    ) -> bool:
        """
        Returns whether a calculation has a result.

        Args:
            scenario_index (Optional[int]): Index of a scenario, if None the current scenario is used.
            calculation_index (Optional[int]): Index of a calculation, if None the current calculation is used.

        Returns:
            bool: Value indicating whether the calculation has a result.
        """
        if calculation_index is None:
            calculation_index = self.current_calculation
        if scenario_index is None:
            scenario_index = self.current_scenario

        return self.datastructure.has_result(scenario_index, calculation_index)

    def get_result(
        self,
        scenario_index: Optional[int] = None,
        calculation_index: Optional[int] = None,
    ) -> DStabilityResult:
        """
        Returns the results of a calculation. Calculation results are based on analysis type and calculation type.

        Args:
            scenario_index (Optional[int]): Index of a scenario, if None is supplied the result of the current scenario is returned.
            calculation_index (Optional[int]): Index of a calculation, if None is supplied the result of the current calculation is returned.

        Returns:
            DStabilityResult: The analysis results of the stage.

        Raises:
            ValueError: No results or calculationsettings available
        """
        if calculation_index is None:
            calculation_index = self.current_calculation
        if scenario_index is None:
            scenario_index = self.current_scenario

        result = self._get_result_substructure(scenario_index, calculation_index)
        return result

    def _get_result_substructure(
        self, scenario_index: Optional[int], calculation_index: Optional[int]
    ) -> DStabilityResult:
        scenario_index = self.get_scenario_index(scenario_index)
        calculation_index = self.get_calculation_index(calculation_index)

        if self.datastructure.has_result(scenario_index, calculation_index):
            result_id = (
                self.datastructure.scenarios[scenario_index]
                .Calculations[calculation_index]
                .ResultId
            )
            calculation_settings = self._get_calculation_settings(
                scenario_index, calculation_index
            )
            analysis_type = calculation_settings.AnalysisType
            calculation_type = calculation_settings.CalculationType

            results = self.datastructure.get_result_substructure(
                analysis_type, calculation_type
            )

            for result in results:
                if result.Id == result_id:
                    return result

        raise ValueError(f"No result found for result id {calculation_index}")

    def _get_calculation_settings(
        self, scenario_index: int, calculation_index: int
    ) -> CalculationSettings:
        calculation_settings_id = (
            self.datastructure.scenarios[scenario_index]
            .Calculations[calculation_index]
            .CalculationSettingsId
        )

        for calculation_settings in self.datastructure.calculationsettings:
            if calculation_settings.Id == calculation_settings_id:
                return calculation_settings

        raise ValueError(
            f"No calculation settings found for calculation {calculation_index} in scenario {scenario_index}."
        )

    def get_slipcircle_result(
        self,
        scenario_index: Optional[int] = None,
        calculation_index: Optional[int] = None,
    ) -> Union[BishopSlipCircleResult, UpliftVanSlipCircleResult]:
        """
        Get the slipcircle(s) of the calculation result of a given stage.

        Args:
            scenario_index (Optional[int]): scenario for which to get the available results
            calculation_index (Optional[int]): calculation for which to get the available results

        Returns:
            Union[BishopSlipCircleResult, UpliftVanSlipCircleResult]: the slipcircle for the given calculation

        Raises:
            ValueError: Result is not available for provided scenario and calculation index
            AttributeError: When the result has no slipcircle. Try get the slipplane
        """
        result = self._get_result_substructure(scenario_index, calculation_index)
        return result.get_slipcircle_output()

    def get_slipplane_result(
        self,
        scenario_index: Optional[int] = None,
        calculation_index: Optional[int] = None,
    ) -> SpencerSlipPlaneResult:
        """
        Get the slipplanes of the calculations result of a calculation.

        Args:
            scenario_index (Optional[int]): scenario for which to get the available results
            calculation_index (Optional[int]): calculation for which to get the available results

        Returns:
            SpencerSlipPlaneResult: the slip plane for the given calculation

        Raises:
            ValueError: Result is not available for provided scenario and calculation index
            AttributeError: When the result has no slipplane. Try get the slipcircle
        """
        result = self._get_result_substructure(scenario_index, calculation_index)
        return result.get_slipplane_output()

    def layer_intersections_at(
        self, x: float
    ) -> List[Tuple[float, float, PersistableSoil]]:
        """Get the intersection with the layers at the given x

        Args:
            x (float): The x coordinate

        Returns:
            List[Tuple[float, float, PersistableSoil]]: A list with top, bottom and soil tuples
        """
        geometry = self._get_geometry(self.current_scenario, self.current_stage)
        return geometry.layer_intersections_at(x, self.layer_soil_dict)

    def _get_geometry(self, scenario_index: int, stage_index: int) -> Geometry:
        geometry_id = (
            self.datastructure.scenarios[scenario_index].Stages[stage_index].GeometryId
        )

        for geometry in self.datastructure.geometries:
            if geometry.Id == geometry_id:
                return geometry

        raise ValueError(
            f"No geometry found for stage {stage_index} in scenario {scenario_index}."
        )

    def _get_soil_layers(self, scenario_index: int, stage_index: int):
        return self.datastructure._get_soil_layers(scenario_index, stage_index)

    def _get_waternet(self, scenario_index: int, stage_index: int):
        waternet_id = (
            self.datastructure.scenarios[scenario_index].Stages[stage_index].WaternetId
        )

        for waternet in self.datastructure.waternets:
            if waternet.Id == waternet_id:
                return waternet

        raise ValueError(
            f"No waternet found for stage {stage_index} in scenario {scenario_index}."
        )

    def _get_waternetcreator_settings(
        self, scenario_index: int, stage_index: int
    ) -> Optional[WaternetCreatorSettings]:
        wncs_id = (
            self.scenarios[scenario_index].Stages[stage_index].WaternetCreatorSettingsId
        )
        for wncs in self.datastructure.waternetcreatorsettings:
            if wncs.Id == wncs_id:
                return wncs

        return None

    def _get_state(self, scenario_index: int, stage_index: int):
        state_id = (
            self.datastructure.scenarios[scenario_index].Stages[stage_index].StateId
        )

        for state in self.datastructure.states:
            if state.Id == state_id:
                return state

        raise ValueError(
            f"No state found for stage {stage_index} in scenario {scenario_index}."
        )

    def _get_state_correlations(self, scenario_index: int, stage_index: int):
        state_correlations_id = (
            self.datastructure.scenarios[scenario_index]
            .Stages[stage_index]
            .StateCorrelationsId
        )

        for state_correlations in self.datastructure.statecorrelations:
            if state_correlations.Id == state_correlations_id:
                return state_correlations

        raise ValueError(
            f"No state correlations found for stage {stage_index} in scenario {scenario_index}."
        )

    def _get_excavations(self, scenario_index: int, stage_index: int):
        return self.datastructure._get_excavations(scenario_index, stage_index)

    def _get_loads(self, scenario_index: int, stage_index: int):
        return self.datastructure._get_loads(scenario_index, stage_index)

    def _get_reinforcements(self, scenario_index: int, stage_index: int):
        reinforcements_id = (
            self.datastructure.scenarios[scenario_index]
            .Stages[stage_index]
            .ReinforcementsId
        )

        for reinforcements in self.datastructure.reinforcements:
            if reinforcements.Id == reinforcements_id:
                return reinforcements

        raise ValueError(
            f"No reinforcements found for stage {stage_index} in scenario {scenario_index}."
        )

    def serialize(self, location: Union[FilePath, DirectoryPath, BinaryIO]):
        """Support serializing to directory while developing for debugging purposes."""
        if isinstance(location, Path) and location.is_dir():
            serializer = DStabilityInputSerializer(ds=self.datastructure)
        else:
            serializer = DStabilityInputZipSerializer(ds=self.datastructure)
        serializer.write(location)
        if isinstance(location, Path):
            self.filename = location

    def add_scenario(
        self, label: str = "Scenario", notes: str = "", set_current: bool = True
    ) -> int:
        """Add a new scenario to the model.

        Args:
            label (str): Label for the scenario.
            notes (str): Notes for the scenario.
            set_current (bool): Whether to make the new scenario the current scenario.

        Returns:
            the id of the new stage
        """
        new_id = self._get_next_id()
        new_scenario_id, new_unique_id = self.datastructure.add_default_scenario(
            label, notes, new_id
        )

        if set_current:
            self.current_scenario = new_scenario_id
            self.current_stage = 0
            self.current_calculation = 0

        self.current_id = new_unique_id
        return new_scenario_id

    def add_stage(
        self,
        scenario_index: Optional[int] = None,
        label: str = "Stage",
        notes: str = "",
        set_current=True,
    ) -> int:
        """Add a new stage to the model at the given scenario index.

        Args:
            scenario_index (Optional[int]): The scenario index to add the stage to, defaults to the current scenario.
            label (str): Label for the stage.
            notes (str): Notes for the stage.
            set_current (bool): Whether to make the new stage the current stage.

        Returns:
            the id of the new stage
        """
        scenario_index = self.get_scenario_index(scenario_index)

        new_id = self._get_next_id()
        new_stage_index, new_unique_id = self.datastructure.add_default_stage(
            scenario_index, label, notes, new_id
        )

        if set_current:
            self.current_stage = new_stage_index
        self.current_id = new_unique_id
        return new_stage_index

    def add_calculation(
        self,
        scenario_index: Optional[int] = None,
        label: str = "Calculation",
        notes: str = "",
        set_current: bool = True,
    ) -> int:
        """Add a new calculation to the model.

        Args:
            scenario_index (Optional[int]): The scenario index to add the calculation to, defaults to the current scenario.
            label (str): Label for the calculation.
            notes (str): Notes for the calculation.
            set_current (bool): Whether to make the new calculation the current calculation.

        Returns:
            the id of the new stage
        """
        scenario_index = self.get_scenario_index(scenario_index)

        new_id = self._get_next_id()
        new_calculation_index, new_unique_id = self.datastructure.add_default_calculation(
            scenario_index, label, notes, new_id
        )

        if set_current:
            self.current_calculation = new_calculation_index
        self.current_id = new_unique_id
        return new_calculation_index

    @property
    def scenarios(self) -> List[Scenario]:
        return self.datastructure.scenarios

    def add_soil(self, soil: Soil) -> int:
        """
        Add a new soil to the model. The code must be unique, the id will be generated

        Args:
            soil (Soil): a new soil

        Returns:
            int: id of the added soil
        """
        if soil.code == None:
            raise ValueError("Soil.code may not be None")
        if self.soils.has_soil_code(soil.code):
            raise ValueError(f"The soil with code {soil.code} is already defined.")

        soil.id = str(self._get_next_id())
        dstability_soil = self.soils.add_soil(soil)
        return int(dstability_soil.Id)

    @property
    def points(self):
        """Enables easy access to the points in the internal dict-like datastructure. Also enables edit/delete for individual points."""

    def add_layer(
        self,
        points: List[Point],
        soil_code: str,
        label: str = "",
        notes: str = "",
        scenario_index: Optional[int] = None,
        stage_index: Optional[int] = None,
    ) -> int:
        """
        Add a soil layer to the model

        Args:
            points (List[Point]): list of Point classes, in clockwise order (non closed simple polygon)
            soil_code (str): code of the soil for this layer
            label (str): label defaults to empty string
            notes (str): notes defaults to empty string
            scenario_index (Optional[int]): scenario to add to, defaults to the current scenario
            stage_index (Optional[int]): stage to add to, defaults to the current stage

        Returns:
            int: id of the added layer
        """
        scenario_index = self.get_scenario_index(scenario_index)
        stage_index = self.get_stage_index(stage_index)

        geometry = self._get_geometry(scenario_index, stage_index)
        soil_layers = self._get_soil_layers(scenario_index, stage_index)

        # Check if we have the soil code
        if not self.soils.has_soil_code(soil_code):
            raise ValueError(
                f"The soil with code {soil_code} is not defined in the soil collection."
            )

        # Make sure the points are valid
        persistable_points = self.make_points_valid(points)

        # Create the new layer
        new_layer = PersistableLayer(
            Id=str(self._get_next_id()),
            Label=label,
            Points=persistable_points,
            Notes=notes,
        )

        # Add the layer to the geometry
        self.add_layer_and_connect_points(geometry.Layers, new_layer)

        # Add the connection between the layer and the soil to soillayers
        soil = self.soils.get_soil(soil_code)
        soil_layers.add_soillayer(layer_id=new_layer.Id, soil_id=soil.Id)
        return int(new_layer.Id)

    # def get_layers(self, x: float) -> Dict:
    #    ls = LineString((x,self.))

    def make_points_valid(self, points: List[Point]) -> List[PersistablePoint]:
        valid_points = make_valid(self.geolib_points_to_shapely_polygon(points))
        return self.to_dstability_points(valid_points)

    def connect_layers(self, layer1: PersistableLayer, layer2: PersistableLayer):
        """Connects two polygons by adding a the missing points on the polygon edges. Returns the two new polygons."""
        linestring1 = self.to_shapely_linestring(layer1.Points)
        linestring2 = self.to_shapely_linestring(layer2.Points)

        # Create a union of the two polygons and polygonize it creating two connected polygons
        union = linestring1.union(linestring2)
        result = [geom for geom in polygonize(union)]

        # If the result has two polygons, we return them, otherwise we return the original polygons
        if len(result) == 2:
            return result[0].exterior, result[1].exterior
        else:
            return linestring1, linestring2

    def add_layer_and_connect_points(
        self, current_layers: List[PersistableLayer], new_layer: PersistableLayer
    ):
        """Adds a new layer to the list of layers and connects the points of the new layer to the existing layers."""

        current_layers.append(new_layer)

        # Check if the new layer intersects with any of the existing layers
        for layer in current_layers:
            if layer != new_layer and self.dstability_points_to_shapely_polygon(
                layer.Points
            ).exterior.intersects(
                self.dstability_points_to_shapely_polygon(new_layer.Points).exterior
            ):
                # If it does, connect the layers
                linestring1, linestring2 = self.connect_layers(layer, new_layer)

                # Update the points of the layers
                current_layers[current_layers.index(layer)].Points = (
                    self.to_dstability_points(linestring1)
                )
                current_layers[current_layers.index(new_layer)].Points = (
                    self.to_dstability_points(linestring2)
                )

    def to_shapely_linestring(self, points: List[PersistablePoint]) -> LineString:
        converted_points = [(p.X, p.Z) for p in points]
        converted_points.append(converted_points[0])
        return LineString(converted_points)

    def dstability_points_to_shapely_polygon(
        self, points: List[PersistablePoint]
    ) -> Polygon:
        return Polygon([(p.X, p.Z) for p in points])

    def geolib_points_to_shapely_polygon(self, points: List[Point]) -> Polygon:
        return Polygon([(p.x, p.z) for p in points])

    def to_dstability_points(
        self, shapely_object: Union[LineString, Polygon]
    ) -> List[PersistablePoint]:
        if isinstance(shapely_object, LineString):
            coords = shapely_object.coords
        elif isinstance(shapely_object, Polygon):
            coords = shapely_object.exterior.coords
        else:
            raise ValueError(
                "shapely_object must be a LineString or Polygon, not {}".format(
                    type(shapely_object)
                )
            )

        persistable_points = [PersistablePoint(X=p[0], Z=p[1]) for p in list(coords)]

        # Remove duplicate points
        persistable_points = [
            i
            for n, i in enumerate(persistable_points)
            if i not in persistable_points[n + 1 :]
        ]

        # Remove last point if it is the same as the first
        if persistable_points[0] == persistable_points[-1]:
            persistable_points.pop(-1)

        return persistable_points

    def get_soil(self, code: str) -> PersistableSoil:
        """
        Gets an existing soil with the given soil code.

        Args:
            code (str): the code of the soil

        Returns:
            PersistableSoil: the soil
        """
        return self.soils.get_soil(code=code)

    def get_soil_by_name(self, name: str) -> PersistableSoil:
        """
        Gets an existing soil with the given soil name.

        Args:
            name (str): the name of the soil

        Returns:
            PersistableSoil: the soil
        """
        return self.soils.get_soil_by_name(name=name)

    def add_head_line(
        self,
        points: List[Point],
        label: str = "",
        notes: str = "",
        is_phreatic_line: bool = False,
        scenario_index: Optional[int] = None,
        stage_index: Optional[int] = None,
    ) -> int:
        """
        Add head line to the model

        Args:
            points (List[Point]): list of Point classes
            label (str): label defaults to empty string
            notes (str): notes defaults to empty string
            is_phreatic_line (bool): set as phreatic line, defaults to False
            scenario_index (Optional[int]): scenario to add to, defaults to the current scenario
            stage_index (Optional[int]): stage to add to, defaults to the current stage

        Returns:
            bool: id of the added headline
        """
        scenario_index = self.get_scenario_index(scenario_index)
        stage_index = self.get_stage_index(stage_index)

        waternet = self._get_waternet(scenario_index, stage_index)

        persistable_headline = waternet.add_head_line(
            str(self._get_next_id()), label, notes, points, is_phreatic_line
        )
        return int(persistable_headline.Id)

    def add_reference_line(
        self,
        points: List[Point],
        bottom_headline_id: int,
        top_head_line_id: int,
        label: str = "",
        notes: str = "",
        scenario_index: Optional[int] = None,
        stage_index: Optional[int] = None,
    ) -> int:
        """
        Add reference line to the model

        Args:
            points (List[Point]): list of Point classes
            bottom_headline_id (int): id of the headline to use as the bottom headline
            top_head_line_id (int): id of the headline to use as the top headline
            label (str): label defaults to empty string
            notes (str): notes defaults to empty string
            scenario_index (Optional[int]): scenario to add to, defaults to the current scenario
            stage_index (Optional[int]): stage to add to, defaults to the current stage

        Returns:
            int: id of the added reference line
        """
        scenario_index = self.get_scenario_index(scenario_index)
        stage_index = self.get_stage_index(stage_index)

        waternet = self._get_waternet(scenario_index, stage_index)

        persistable_reference_line = waternet.add_reference_line(
            str(self._get_next_id()),
            label,
            notes,
            points,
            str(bottom_headline_id),
            str(top_head_line_id),
        )
        return int(persistable_reference_line.Id)

    def add_state_point(
        self,
        state_point: DStabilityStatePoint,
        scenario_index: Optional[int] = None,
        stage_index: Optional[int] = None,
    ) -> int:
        """
        Add state point to the model

        Args:
            state_point (DStabilityStatePoint): DStabilityStatePoint class
            scenario_index (Optional[int]): scenario to add to, defaults to the current scenario
            stage_index (Optional[int]): stage to add to, defaults to the current stage

        Returns:
            int: id of the added add_state_point
        """
        scenario_index = self.get_scenario_index(scenario_index)
        stage_index = self.get_stage_index(stage_index)

        states = self._get_state(scenario_index, stage_index)

        try:
            _ = self._get_geometry(scenario_index, stage_index).get_layer(
                state_point.layer_id
            )
        except ValueError:
            raise ValueError(f"No layer with id '{state_point.layer_id} in this geometry")

        state_point.id = (
            self._get_next_id()
        )  # the user does not know the id so we have to add it
        persistable_state_point = state_point._to_internal_datastructure()
        states.add_state_point(persistable_state_point)
        return int(persistable_state_point.Id)

    def add_state_line(
        self,
        points: List[Point],
        state_points: List[DStabilityStateLinePoint],
        scenario_index: Optional[int] = None,
        stage_index: Optional[int] = None,
    ) -> int:
        """
        Add state line. From the Soils, only the state parameters are used.

        points are a list of points with x,z coordinates
        state_point are a list of DStabilityStateLinePoint where ONLY the x is used, the Z will be calculated

        Args:
            points (List[Point]): The geometry points of the state line.
            state_point (List[DStabilityStatePoint]): The list of state point values.
            scenario_index (Optional[int]): scenario to add to, defaults to the current scenario.
            stage_index (Optional[int]): stage to add to, defaults to the current stage.

        Returns:
            PersistableStateLine: The created state line
        """
        scenario_index = self.get_scenario_index(scenario_index)
        stage_index = self.get_stage_index(stage_index)

        states = self._get_state(scenario_index, stage_index)

        # each point should belong to a layer
        persistable_points = []

        for point in points:
            point.id = self._get_next_id()  # assign a new id
            persistable_points.append(PersistablePoint(X=point.x, Z=point.z))

        persistable_state_line_points = []
        for state_point in state_points:
            state_point.id = self._get_next_id()  # assign a new id
            persistable_state_line_points.append(state_point._to_internal_datastructure())

        return states.add_state_line(persistable_points, persistable_state_line_points)

    def add_state_correlation(
        self,
        correlated_state_ids: List[int],
        scenario_index: Optional[int] = None,
        stage_index: Optional[int] = None,
    ):
        """
        Add state correlation between the given state point ids.

        Args:
            correlated_state_ids (List[int]): The state point ids to correlate.
            scenario_index (Optional[int]): scenario to add to, defaults to the current scenario.
            stage_index (Optional[int]): stage to add to, defaults to the current stage.

        Returns:
            None
        """
        scenario_index = self.get_scenario_index(scenario_index)
        stage_index = self.get_stage_index(stage_index)

        state_correlations = self._get_state_correlations(scenario_index, stage_index)

        for state_id in correlated_state_ids:
            try:
                _ = self._get_state(scenario_index, stage_index).get_state(state_id)
            except ValueError:
                raise ValueError(f"No state point with id '{state_id} in this geometry")

        persistable_state_correlation = PersistableStateCorrelation(
            CorrelatedStateIds=correlated_state_ids, IsFullyCorrelated=True
        )

        state_correlations.add_state_correlation(persistable_state_correlation)

    def add_excavation(
        self,
        points: List[Point],
        label: str,
        notes: str = "",
        scenario_index: Optional[int] = None,
        stage_index: Optional[int] = None,
    ):
        scenario_index = self.get_scenario_index(scenario_index)
        stage_index = self.get_stage_index(stage_index)

        persistable_excavation = PersistableExcavation(
            Label=label,
            Notes=notes,
            Points=[PersistablePoint(X=p.x, Z=p.z) for p in points],
        )
        self._get_excavations(scenario_index, stage_index).append(persistable_excavation)

    def add_load(
        self,
        load: DStabilityLoad,
        consolidations: Optional[List[Consolidation]] = None,
        scenario_index: Optional[int] = None,
        stage_index: Optional[int] = None,
    ) -> None:
        """Add a load to the object.

        The geometry should be defined before adding loads.

        If no consolidations are provided, a Consolidation with default values will be made for each SoilLayer.
        It is not possible to set consolidation degrees of loads afterwards since they don't have an id.

        Args:
            load: A subclass of DStabilityLoad.
            scenario_index (Optional[int]): scenario to add to, defaults to the current scenario.
            stage_index (Optional[int]): stage to add to, defaults to the current stage.

        Raises:
            ValueError: When the provided load is no subclass of DStabilityLoad, an invalid stage_index is provided, or the datastructure is no longer valid.
        """
        scenario_index = self.get_scenario_index(scenario_index)
        stage_index = self.get_stage_index(stage_index)

        if not issubclass(type(load), DStabilityLoad):
            raise ValueError(
                f"load should be a subclass of DstabilityReinforcement, received {load}"
            )
        if self.datastructure.has_soil_layers(
            scenario_index, stage_index
        ) and self.datastructure.has_loads(scenario_index, stage_index):
            if consolidations is None:
                consolidations = self._get_default_consolidations(
                    scenario_index, stage_index
                )
            else:
                self._verify_consolidations(consolidations, scenario_index, stage_index)
            self._get_loads(scenario_index, stage_index).add_load(load, consolidations)
        else:
            raise ValueError(
                f"No loads found for scenario {scenario_index} stage {stage_index}"
            )

    def add_soil_layer_consolidations(
        self,
        soil_layer_id: int,
        consolidations: Optional[List[Consolidation]] = None,
        scenario_index: Optional[int] = None,
        stage_index: Optional[int] = None,
    ) -> None:
        """Add consolidations for a layer (layerload).

        Consolidations cannot be added when adding soil layers since in the consolidations, all other soil layers need to be referred.
        Therefore, all soillayers in a stage should be defined before setting consolidation and
        the number of consolidations given should equal the amount of layers.

        Args:
            soil_layer_id: Consolidation is set for this soil layer id.
            consolidations: List of Consolidation. Must contain a Consolidation for every other layer.
            scenario_index (Optional[int]): scenario to add to, defaults to the current scenario.
            stage_index (Optional[int]): stage to add to, defaults to the current stage.

        Raises:
            ValueError: When the provided load is no subclass of DStabilityLoad, an invalid stage_index is provided, or the datastructure is no longer valid.
        """
        scenario_index = self.get_scenario_index(scenario_index)
        stage_index = self.get_stage_index(stage_index)

        if self.datastructure.has_soil_layer(
            scenario_index, stage_index, soil_layer_id
        ) and self.datastructure.has_loads(scenario_index, stage_index):
            if consolidations is None:
                consolidations = self._get_default_consolidations(
                    scenario_index, stage_index, soil_layer_id
                )
            else:
                self._verify_consolidations(
                    consolidations, scenario_index, stage_index, soil_layer_id
                )

            self._get_loads(scenario_index, stage_index).add_layer_load(
                soil_layer_id, consolidations
            )
        else:
            raise ValueError(
                f"No soil layer loads found for scenario {scenario_index} stage {stage_index}"
            )

    def _get_default_consolidations(
        self,
        scenario_index: int,
        stage_index: int,
        exclude_soil_layer_id: Optional[int] = None,
    ) -> List[Consolidation]:
        """Length of the consolidations is equal to the amount of soil layers.

        If exclude_soil_layer_id is provided, that specific soil layer id is not included in the consolidations.
        """
        if self.datastructure.has_soil_layers(scenario_index, stage_index):
            soil_layer_ids = self._get_soil_layers(scenario_index, stage_index).get_ids(
                exclude_soil_layer_id
            )
            return [Consolidation(layer_id=layer_id) for layer_id in soil_layer_ids]

        raise ValueError(f"No soil layers found for stage at index {stage_index}")

    def _verify_consolidations(
        self,
        consolidations: List[Consolidation],
        scenario_index: int,
        stage_index: int,
        exclude_soil_layer_id: Optional[int] = None,
    ) -> None:
        if self.datastructure.has_soil_layers(scenario_index, stage_index):
            consolidation_soil_layer_ids: Set[str] = {
                str(c.layer_id) for c in consolidations
            }
            soil_layer_ids = self._get_soil_layers(scenario_index, stage_index).get_ids(
                exclude_soil_layer_id
            )

            if consolidation_soil_layer_ids != soil_layer_ids:
                raise ValueError(
                    f"Received consolidations ({consolidation_soil_layer_ids}) should contain all soil layer ids ({soil_layer_ids})"
                )
        else:
            raise ValueError(f"No soil layers found for stage at index {stage_index}")

    def add_reinforcement(
        self,
        reinforcement: DStabilityReinforcement,
        scenario_index: Optional[int] = None,
        stage_index: Optional[int] = None,
    ) -> None:
        """Add a reinforcement to the model.

        Args:
            reinforcement: A subclass of DStabilityReinforcement.
            scenario_index (Optional[int]): scenario to add to, defaults to the current scenario.
            stage_index (Optional[int]): stage to add to, defaults to the current stage.

        Returns:
            int: Assigned id of the reinforcements (collection object of all reinforcements for a stage).

        Raises:
            ValueError: When the provided reinforcement is no subclass of DStabilityReinforcement, an invalid stage_index is provided, or the datastructure is no longer valid.
        """
        scenario_index = self.get_scenario_index(scenario_index)
        stage_index = self.get_stage_index(stage_index)

        if not issubclass(type(reinforcement), DStabilityReinforcement):
            raise ValueError(
                f"reinforcement should be a subclass of DstabilityReinforcement, received {reinforcement}"
            )

        if self.datastructure.has_reinforcements(scenario_index, stage_index):
            self._get_reinforcements(scenario_index, stage_index).add_reinforcement(
                reinforcement
            )
        else:
            raise ValueError(
                f"No reinforcements found for scenario {scenario_index} stage {stage_index}"
            )

    def add_soil_correlation(self, list_correlated_soil_ids: List[str]):
        """Add a soil correlation to the model.

        Args:
            list_correlated_soil_ids: A list of soil ids that are correlated.
        """
        self.soil_correlations.add_soil_correlation(list_correlated_soil_ids)

    def set_model(
        self,
        analysis_method: DStabilityAnalysisMethod,
        scenario_index: Optional[int] = None,
        calculation_index: Optional[int] = None,
    ) -> None:
        """Sets the model and applies the given parameters

        Args:
            analysis_method (DStabilityAnalysisMethod): A subclass of DStabilityAnalysisMethod.
            scenario_index (Optional[int]): scenario to add to, defaults to the current scenario
            calculation_index (Optional[int]): calculation to add to, defaults to the current calculation

        Raises:
            ValueError: When the provided analysis method is no subclass of DStabilityAnalysisMethod,
            an invalid stage_index is provided, the analysis method is not known or the datastructure is no longer valid.
        """
        scenario_index = self.get_scenario_index(scenario_index)
        calculation_index = self.get_calculation_index(calculation_index)

        calculationsettings = self._get_calculation_settings(
            scenario_index, calculation_index
        )

        _analysis_method_mapping = {
            AnalysisType.BISHOP: calculationsettings.set_bishop,
            AnalysisType.BISHOP_BRUTE_FORCE: calculationsettings.set_bishop_brute_force,
            AnalysisType.SPENCER: calculationsettings.set_spencer,
            AnalysisType.SPENCER_GENETIC: calculationsettings.set_spencer_genetic,
            AnalysisType.UPLIFT_VAN: calculationsettings.set_uplift_van,
            AnalysisType.UPLIFT_VAN_PARTICLE_SWARM: calculationsettings.set_uplift_van_particle_swarm,
        }

        try:
            _analysis_method_mapping[analysis_method.analysis_type](
                analysis_method._to_internal_datastructure()
            )
        except KeyError:
            raise ValueError(
                f"Unknown analysis method {analysis_method.analysis_type.value} found"
            )

    def get_scenario_index(self, scenario_index: Optional[int]):
        if scenario_index is None:
            return self.current_scenario
        else:
            return scenario_index

    def get_stage_index(self, stage_index: Optional[int]):
        if stage_index is None:
            return self.current_stage
        else:
            return stage_index

    def get_calculation_index(self, calculation_index: Optional[int]):
        if calculation_index is None:
            return self.current_calculation
        else:
            return calculation_index

    @staticmethod
    def get_soil_id_from_layer_id(
        layers: SoilLayerCollection, layer_id: str
    ) -> Union[str, None]:
        for layer in layers.SoilLayers:
            if layer.LayerId == layer_id:
                return layer.SoilId
        return None

    @staticmethod
    def get_color_from_soil_id(
        soil_visualizations: SoilVisualisation, soil_id: str
    ) -> str:
        for soil_visualization in soil_visualizations.SoilVisualizations:
            if soil_visualization.SoilId == soil_id:
                return soil_visualization.Color
        return "#000000"

    def _get_color_of_layer(
        self, layers_collection: SoilLayerCollection, layer: PersistableLayer
    ) -> str:
        layer_id = layer.Id
        # use the layer id to get the soil type id
        soil_type_id = DStabilityModel.get_soil_id_from_layer_id(
            layers_collection, layer_id
        )
        # get the color of the soil type
        color = DStabilityModel.get_color_from_soil_id(
            self.input.soilvisualizations, soil_type_id
        )
        return color.replace("#80", "#")

    def plot(
        self, scenario_index: Optional[int] = None, stage_index: Optional[int] = None
    ):
        geometry = self._get_geometry(scenario_index, stage_index)
        layers_collection = self._get_soil_layers(scenario_index, stage_index)
        fig, ax = plt.subplots()
        # loop over the layers
        for layer in geometry.Layers:
            # get list of x and y coordinates
            x = [p.X for p in layer.Points]
            y = [p.Z for p in layer.Points]
            # get color of layer
            color = self._get_color_of_layer(layers_collection, layer)
            # create a polygon
            ax.fill(x, y, color=color)
        plt.axis("off")
        return fig, ax
