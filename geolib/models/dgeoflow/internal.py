"""
The internal data model structure.
"""

from collections import defaultdict
from datetime import date, datetime
from enum import Enum
from itertools import chain
from math import isfinite
from typing import Dict, List, Optional, Set, Tuple, Union

from geolib._compat import IS_PYDANTIC_V2

if IS_PYDANTIC_V2:
    from pydantic import ConfigDict, Field, field_validator, model_validator
    from typing_extensions import Annotated
else:
    from pydantic import conlist, root_validator, validator

from geolib import BaseModelStructure
from geolib import __version__ as version
from geolib.geometry import Point
from geolib.soils import Soil, StorageParameters
from geolib.utils import snake_to_camel

from .dgeoflow_validator import DGeoFlowValidator
from .utils import children


class DGeoFlowBaseModelStructure(BaseModelStructure):
    def dict(_, *args, **kwargs):
        if IS_PYDANTIC_V2:
            data = super().model_dump(*args, **kwargs)
        else:
            data = super().dict(*args, **kwargs)
        return {
            k: "NaN" if isinstance(v, float) and not isfinite(v) else v
            for k, v in data.items()
        }

    @classmethod
    def transform_id_to_str(cls, value) -> str:
        if value is None:
            return None
        return str(value)


class DGeoFlowSubStructure(DGeoFlowBaseModelStructure):
    @classmethod
    def structure_name(cls):
        class_name = cls.__name__
        return str.split(str.lower(class_name), ".")[-1]

    @classmethod
    def structure_group(cls):
        return cls.structure_name()


class CalculationTypeEnum(Enum):
    GROUNDWATER_FLOW = "GroundwaterFlow"
    PIPE_LENGTH = "PipeLength"
    CRITICAL_HEAD = "CriticalHead"


CalculationType = CalculationTypeEnum


class PersistableStochasticParameter(DGeoFlowBaseModelStructure):
    IsProbabilistic: bool = False
    Mean: float = 1.0
    StandardDeviation: float = 0.0


class PersistableShadingTypeEnum(Enum):
    DIAGONAL_A = "DiagonalA"
    DIAGONAL_B = "DiagonalB"
    DIAGONAL_C = "DiagonalC"
    DIAGONAL_D = "DiagonalD"
    DOT_A = "DotA"
    DOT_B = "DotB"
    DOT_C = "DotC"
    DOT_D = "DotD"
    HORIZONTAL_A = "HorizontalA"
    HORIZONTAL_B = "HorizontalB"
    NONE = "None"


class PersistableSoilVisualization(DGeoFlowBaseModelStructure):
    Color: Optional[str]
    PersistableShadingType: Optional[PersistableShadingTypeEnum]
    SoilId: Optional[str]

    if IS_PYDANTIC_V2:
        id_validator = field_validator("SoilId", mode="before")(
            DGeoFlowBaseModelStructure.transform_id_to_str
        )


class SoilVisualisation(DGeoFlowBaseModelStructure):
    ContentVersion: Optional[str] = "2"
    SoilVisualizations: Optional[List[Optional[PersistableSoilVisualization]]] = []

    @classmethod
    def structure_name(cls) -> str:
        return "soilvisualizations"


class PersistableSoilLayer(DGeoFlowBaseModelStructure):
    LayerId: Optional[str] = None
    SoilId: Optional[str] = None

    if IS_PYDANTIC_V2:
        id_validator = field_validator("LayerId", "SoilId", mode="before")(
            DGeoFlowBaseModelStructure.transform_id_to_str
        )


class SoilLayerCollection(DGeoFlowSubStructure):
    """soillayers/soillayers_x.json"""

    @classmethod
    def structure_name(cls) -> str:
        return "soillayers"

    @classmethod
    def structure_group(cls) -> str:
        return "soillayers"

    ContentVersion: Optional[str] = "2"
    Id: Optional[str] = None
    SoilLayers: List[PersistableSoilLayer] = []

    if IS_PYDANTIC_V2:
        id_validator = field_validator("Id", mode="before")(
            DGeoFlowBaseModelStructure.transform_id_to_str
        )

    def add_soillayer(self, layer_id: str, soil_id: str) -> PersistableSoilLayer:
        psl = PersistableSoilLayer(LayerId=layer_id, SoilId=soil_id)
        self.SoilLayers.append(psl)
        return psl

    def get_ids(self, exclude_soil_layer_id: Optional[int]) -> Set[str]:
        if exclude_soil_layer_id is not None:
            exclude_soil_layer_id = str(exclude_soil_layer_id)
        return {
            layer.LayerId
            for layer in self.SoilLayers
            if layer.LayerId != exclude_soil_layer_id
        }


class PersistableSoil(DGeoFlowBaseModelStructure):
    Code: str = ""
    Id: str = ""
    Name: str = ""
    Notes: str = ""
    HorizontalPermeability: float = 0.001
    VerticalPermeability: float = 0.001

    if IS_PYDANTIC_V2:
        id_validator = field_validator("Id", mode="before")(
            DGeoFlowBaseModelStructure.transform_id_to_str
        )


class SoilCollection(DGeoFlowSubStructure):
    """soils.json"""

    ContentVersion: Optional[str] = "2"
    Soils: List[PersistableSoil] = [
        PersistableSoil(
            Code="H_Aa_ht_new",
            Id="2",
            Name="Embankment new",
            HorizontalPermeability=0.01,
            VerticalPermeability=0.01,
        ),
        PersistableSoil(
            Id="3",
            Name="Embankment old",
            Code="H_Aa_ht_old",
            HorizontalPermeability=0.01,
            VerticalPermeability=0.01,
        ),
        PersistableSoil(
            Id="4",
            Name="Clay, shallow",
            Code="H_Rk_k_shallow",
            HorizontalPermeability=0.01,
            VerticalPermeability=0.01,
        ),
        PersistableSoil(
            Id="5",
            Name="Clay, deep",
            Code="H_Rk_k_deep",
            HorizontalPermeability=0.01,
            VerticalPermeability=0.01,
        ),
        PersistableSoil(
            Id="6",
            Name="Organic clay",
            Code="H_Rk_ko",
            HorizontalPermeability=0.01,
            VerticalPermeability=0.01,
        ),
        PersistableSoil(
            Id="7",
            Name="Peat, shallow",
            Code="H_vhv_v",
            HorizontalPermeability=0.01,
            VerticalPermeability=0.01,
        ),
        PersistableSoil(
            Id="8",
            Name="Peat, deep",
            Code="H_vbv_v",
            HorizontalPermeability=0.01,
            VerticalPermeability=0.01,
        ),
        PersistableSoil(
            Id="9",
            Name="Sand",
            Code="Sand",
            HorizontalPermeability=30,
            VerticalPermeability=30,
        ),
        PersistableSoil(
            Id="10",
            Name="Clay with silt",
            Code="P_Rk_k&s",
            HorizontalPermeability=0.1,
            VerticalPermeability=0.1,
        ),
        PersistableSoil(
            Id="11",
            Name="Sand with clay",
            Code="H_Ro_z&k",
            HorizontalPermeability=1,
            VerticalPermeability=1,
        ),
        PersistableSoil(
            Id="12",
            Name="Sand, less permeable",
            Code="Sand, less permeable",
            HorizontalPermeability=15,
            VerticalPermeability=15,
        ),
        PersistableSoil(
            Id="13",
            Name="Sand, permeable",
            Code="Sand, permeable",
            HorizontalPermeability=45,
            VerticalPermeability=45,
        ),
    ]

    @classmethod
    def structure_name(cls) -> str:
        return "soils"

    def has_soilcode(self, code: str) -> bool:
        """
        Checks if the soilcode is available in the current soil list.

        Args:
            code (str): code of the soil

        Returns:
            bool: True if found, False if not
        """
        return code in {s.Code for s in self.Soils}

    def add_soil(self, soil: Soil) -> PersistableSoil:
        """
        Add a new soil to the model.

        Args:
            soil (Soil): a new soil

        Returns:
            None
        """
        ps = soil._to_dgeoflow()

        self.Soils.append(ps)
        return ps

    def __internal_soil_to_global_soil(self, persistable_soil: PersistableSoil):
        storage_parameters = StorageParameters(
            vertical_permeability=persistable_soil.VerticalPermeability,
            horizontal_permeability=persistable_soil.HorizontalPermeability,
        )

        return Soil(
            id=persistable_soil.Id,
            name=persistable_soil.Name,
            code=persistable_soil.Code,
            storage_parameters=storage_parameters,
        )

    def get_soil(self, code: str) -> Soil:
        """
        Get soil by the given code.

        Args:
            code (str): code of the soil

        Returns:
            Soil: the soil object
        """
        for persistable_soil in self.Soils:
            if persistable_soil.Code == code:
                return self.__internal_soil_to_global_soil(persistable_soil)

        raise ValueError(f"Soil code '{code}' not found in the SoilCollection")

    def edit_soil(self, code: str, **kwargs: dict) -> PersistableSoil:
        """
        Update a soil.

        Args:
            code (str): code of the soil.
            kwargs (dict): dictionary with argument names and values

        Returns:
            PersistableSoil: the edited soil
        """

        for persistable_soil in self.Soils:
            if persistable_soil.Code == code:
                return self.edit_persistable_soil(
                    persistable_soil=persistable_soil, kwargs=kwargs
                )
        raise ValueError(f"Soil code '{code}' not found in the SoilCollection")

    def edit_soil_by_name(
        self, name: Optional[str] = None, **kwargs: dict
    ) -> PersistableSoil:
        """
        Update a soil, searching by name. This method will edit the first occurence of the name
        if it is used multiple times.

        Args:
            name (str): name of the soil.
            kwargs (dict): dictionary with argument names and values

        Returns:
            PersistableSoil: the edited soil
        """

        for persistable_soil in self.Soils:
            if persistable_soil.Name == name:
                return self.edit_persistable_soil(
                    persistable_soil=persistable_soil, kwargs=kwargs
                )
        raise ValueError(f"Soil name '{name}' not found in the SoilCollection")

    def edit_persistable_soil(self, persistable_soil: PersistableSoil, kwargs: dict):
        for k, v in kwargs.items():
            try:
                setattr(persistable_soil, snake_to_camel(k), v)
                k_stochastic = f"{snake_to_camel(k)}StochasticParameter"
                if hasattr(persistable_soil, k_stochastic):
                    getattr(persistable_soil, k_stochastic).Mean = v
            except AttributeError:
                raise ValueError(f"Unknown soil parameter {k}.")
        return persistable_soil


class ProjectInfo(DGeoFlowSubStructure):
    """projectinfo.json."""

    Analyst: Optional[str] = ""
    ApplicationCreated: Optional[str] = ""
    ApplicationModified: Optional[str] = ""
    ContentVersion: Optional[str] = "2"
    Created: Optional[date] = datetime.now().date()
    CrossSection: Optional[str] = ""
    Date: Optional[date] = datetime.now().date()
    IsDataValidated: Optional[bool] = False
    LastModified: Optional[date] = datetime.now().date()
    LastModifier: Optional[str] = "GEOLib"
    Path: Optional[str] = ""
    Project: Optional[str] = ""
    Remarks: Optional[str] = f"Created with GEOLib {version}"

    @classmethod
    def nltime(cls, date: Union[date, str]) -> date:
        if isinstance(date, str):
            position = date.index(max(date.split("-"), key=len))
            if position > 0:
                date = datetime.strptime(date, "%d-%m-%Y").date()
            else:
                date = datetime.strptime(date, "%Y-%m-%d").date()
        return date

    if IS_PYDANTIC_V2:
        nltime_validator = field_validator(
            "Created", "Date", "LastModified", mode="before"
        )(nltime)
    else:
        nltime_validator = validator(
            "Created", "Date", "LastModified", pre=True, allow_reuse=True
        )(nltime)


class PersistablePoint(DGeoFlowBaseModelStructure):
    X: Optional[float] = 0
    Z: Optional[float] = 0


class PersistableLayer(DGeoFlowBaseModelStructure):
    Id: Optional[str] = None
    Label: Optional[str] = None
    Notes: Optional[str] = None
    if IS_PYDANTIC_V2:
        Points: Annotated[List[PersistablePoint], Field(min_length=3)]
    else:
        Points: conlist(PersistablePoint, min_items=3)

    if IS_PYDANTIC_V2:
        id_validator = field_validator("Id", mode="before")(
            DGeoFlowBaseModelStructure.transform_id_to_str
        )

    @classmethod
    def polygon_checks(cls, points):
        """
        Todo:
            Find a way to check the validity of the given points
        """
        # implement some checks
        # 1. is this a simple polygon
        # 2. is it clockwise
        # 3. is it a non closed polygon
        # 4. does it intersect other polygons
        return points

    if IS_PYDANTIC_V2:
        polygon_checks_validator = field_validator("Points", mode="before")(
            polygon_checks
        )
    else:
        polygon_checks_validator = validator("Points", pre=True, allow_reuse=True)(
            polygon_checks
        )


class Geometry(DGeoFlowSubStructure):
    """geometries/geometry_x.json"""

    @classmethod
    def structure_group(cls) -> str:
        return "geometries"

    @classmethod
    def structure_name(cls) -> str:
        return "geometry"

    ContentVersion: Optional[str] = "2"
    Id: Optional[str] = None
    Layers: List[PersistableLayer] = []

    if IS_PYDANTIC_V2:
        id_validator = field_validator("Id", mode="before")(
            DGeoFlowBaseModelStructure.transform_id_to_str
        )

    def contains_point(self, point: Point) -> bool:
        """
        Check if the given point is on one of the points of the layers

        Args:
            point (Point): A point type

        Returns:
            bool: True if this point is found on a layer, False otherwise

        Todo:
            Take x, z accuracy into account
        """
        for layer in self.Layers:
            for p in layer.Points:
                if point.x == p.X and point.z == p.Z:  # not nice
                    return True

        return False

    def get_layer(self, id: int) -> PersistableLayer:
        for layer in self.Layers:
            if layer.Id == str(id):
                return layer

        raise ValueError(f"Layer id {id} not found in this geometry")

    def add_layer(
        self, id: str, label: str, notes: str, points: List[Point]
    ) -> PersistableLayer:
        """
        Add a new layer to the model. Layers are expected;
        1. to contain at least 3 point (non closed polygons)
        2. the points need to be in clockwise order
        3. the polygon needs to be convex (no intersections with itsself)

        Args:
            id (str): id of the layer
            label (str): label of the layer
            notes (str): notes for the layers
            points (List[Points]): list of Point classes

        Returns:
            PersistableLayer: the layer as a persistable object
        """
        layer = PersistableLayer(
            Id=id,
            Label=label,
            Notes=notes,
            Points=[PersistablePoint(X=p.x, Z=p.z) for p in points],
        )

        self.Layers.append(layer)
        return layer


class PersistableFixedHeadBoundaryConditionProperties(DGeoFlowBaseModelStructure):
    HeadLevel: float


class PersistableBoundaryCondition(DGeoFlowBaseModelStructure):
    Label: Optional[str] = None
    Notes: Optional[str] = None
    Id: Optional[str] = None
    if IS_PYDANTIC_V2:
        Points: Annotated[List[PersistablePoint], Field(min_length=2)]
    else:
        Points: conlist(PersistablePoint, min_items=2)
    FixedHeadBoundaryConditionProperties: PersistableFixedHeadBoundaryConditionProperties

    if IS_PYDANTIC_V2:
        id_validator = field_validator("Id", mode="before")(
            DGeoFlowBaseModelStructure.transform_id_to_str
        )


class BoundaryConditionCollection(DGeoFlowSubStructure):
    """boundaryconditions/boundaryconditions_x.json"""

    ContentVersion: Optional[str] = "2"
    Id: Optional[str] = None
    BoundaryConditions: List[PersistableBoundaryCondition] = []

    if IS_PYDANTIC_V2:
        id_validator = field_validator("Id", mode="before")(
            DGeoFlowBaseModelStructure.transform_id_to_str
        )

    @classmethod
    def structure_group(cls) -> str:
        return "boundaryconditions"

    @classmethod
    def structure_name(cls) -> str:
        return "boundaryconditions"

    def contains_point(self, point: Point) -> bool:
        """
        Check if the given point is on one of the points of the layers

        Args:
            point (Point): A point type

        Returns:
            bool: True if this point is found on a layer, False otherwise

        Todo:
            Take x, z accuracy into account
        """
        for layer in self.Layers:
            for p in layer.Points:
                if point.x == p.X and point.z == p.Z:  # not nice
                    return True

        return False

    def add_boundary_condition(
        self, id: int, label: str, notes: str, points: List[Point], head_level: float
    ) -> PersistableBoundaryCondition:
        pbc_properties = PersistableFixedHeadBoundaryConditionProperties(
            HeadLevel=head_level
        )
        pbc = PersistableBoundaryCondition(
            Id=str(id),
            Label=label,
            Notes=notes,
            Points=[PersistablePoint(X=p.x, Z=p.z) for p in points],
            FixedHeadBoundaryConditionProperties=pbc_properties,
        )
        self.BoundaryConditions.append(pbc)
        return pbc


class PersistableStage(DGeoFlowBaseModelStructure):
    Label: Optional[str] = None
    Notes: Optional[str] = None
    BoundaryConditionCollectionId: Optional[str] = None


class ErosionDirectionEnum(Enum):
    LEFT_TO_RIGHT = ("LeftToRight",)
    RIGHT_TO_LEFT = "RightToLeft"


class InternalPipeTrajectory(DGeoFlowBaseModelStructure):
    Label: Optional[str] = None
    Notes: Optional[str] = None
    D70: Optional[float] = None
    Points: Optional[List[PersistablePoint]] = None
    ErosionDirection: Optional[ErosionDirectionEnum] = ErosionDirectionEnum.RIGHT_TO_LEFT
    ElementSize: Optional[float] = None


class PersistableCriticalHeadSearchSpace(DGeoFlowBaseModelStructure):
    MinimumHeadLevel: Optional[float] = 0
    MaximumHeadLevel: Optional[float] = 1
    StepSize: Optional[float] = 0.1


class PersistableCalculation(DGeoFlowBaseModelStructure):
    Label: Optional[str] = None
    Notes: Optional[str] = None
    CalculationType: Optional[CalculationTypeEnum] = CalculationTypeEnum.GROUNDWATER_FLOW
    CriticalHeadId: Optional[str] = None
    CriticalHeadSearchSpace: Optional[
        PersistableCriticalHeadSearchSpace
    ] = PersistableCriticalHeadSearchSpace()
    PipeTrajectory: Optional[InternalPipeTrajectory] = None
    MeshPropertiesId: Optional[str] = None
    ResultsId: Optional[str] = None

    if IS_PYDANTIC_V2:
        id_validator = field_validator(
            "CriticalHeadId", "MeshPropertiesId", "ResultsId", mode="before"
        )(DGeoFlowBaseModelStructure.transform_id_to_str)


class NodeResult(DGeoFlowBaseModelStructure):
    Point: Optional[PersistablePoint] = None
    TotalPorePressure: float = 1
    HydraulicDischarge: float = 1
    HydraulicHead: float = 1


class ElementResult(DGeoFlowBaseModelStructure):
    NodeResults: Optional[List[NodeResult]] = []


class PipeElementResult(DGeoFlowBaseModelStructure):
    Nodes: Optional[List[PersistablePoint]] = []
    IsActive: Optional[bool] = None
    Height: Optional[float] = None


class GroundwaterFlowResult(DGeoFlowSubStructure):
    Id: Optional[str] = None
    Elements: Optional[List[ElementResult]] = []
    ContentVersion: Optional[str] = "2"

    if IS_PYDANTIC_V2:
        id_validator = field_validator("Id", mode="before")(
            DGeoFlowBaseModelStructure.transform_id_to_str
        )

    @classmethod
    def structure_group(cls) -> str:
        return "results/groundwaterflow/"


class PipeLengthResult(DGeoFlowSubStructure):
    Id: Optional[str] = None
    PipeLength: Optional[float] = None
    Elements: Optional[List[ElementResult]] = []
    PipeElements: Optional[List[PipeElementResult]] = []
    ContentVersion: Optional[str] = "2"

    if IS_PYDANTIC_V2:
        id_validator = field_validator("Id", mode="before")(
            DGeoFlowBaseModelStructure.transform_id_to_str
        )

    @classmethod
    def structure_group(cls) -> str:
        return "results/pipelength/"


class CriticalHeadResult(DGeoFlowSubStructure):
    Id: Optional[str] = None
    PipeLength: Optional[float] = None
    CriticalHead: Optional[float] = None
    Elements: Optional[List[ElementResult]] = []
    PipeElements: Optional[List[PipeElementResult]] = []
    ContentVersion: Optional[str] = "2"

    if IS_PYDANTIC_V2:
        id_validator = field_validator("Id", mode="before")(
            DGeoFlowBaseModelStructure.transform_id_to_str
        )

    @classmethod
    def structure_group(cls) -> str:
        return "results/criticalhead/"


DGeoFlowResult = Union[GroundwaterFlowResult, PipeLengthResult, CriticalHeadResult]


class Scenario(DGeoFlowSubStructure):
    """scenarios/scenario_x.json"""

    ContentVersion: Optional[str] = "2"
    Id: Optional[str] = None
    Label: Optional[str] = None
    Notes: Optional[str] = None
    GeometryId: Optional[str] = None
    SoilLayersId: Optional[str] = None
    Stages: List[PersistableStage] = []
    Calculations: List[PersistableCalculation] = []

    if IS_PYDANTIC_V2:
        id_validator = field_validator("Id", "GeometryId", "SoilLayersId", mode="before")(
            DGeoFlowBaseModelStructure.transform_id_to_str
        )

    @classmethod
    def structure_name(cls) -> str:
        return "scenario"

    @classmethod
    def structure_group(cls) -> str:
        return "scenarios"

    def add_calculation(
        self, label: str, notes: str, mesh_properties_id: str
    ) -> PersistableCalculation:
        pc = PersistableCalculation(
            Label=label, Notes=notes, MeshPropertiesId=mesh_properties_id
        )
        self.Calculations.append(pc)
        return pc

    def add_stage(
        self, label: str, notes: str, boundaryconditions_collection_id: str
    ) -> PersistableStage:
        ps = PersistableStage(
            Label=label,
            Notes=notes,
            BoundaryConditionCollectionId=boundaryconditions_collection_id,
        )
        self.Stages.append(ps)
        return ps


class PersistableMeshProperties(DGeoFlowBaseModelStructure):
    LayerId: str
    Label: Optional[str] = None
    ElementSize: Optional[float] = 1

    if IS_PYDANTIC_V2:
        id_validator = field_validator("LayerId", mode="before")(
            DGeoFlowBaseModelStructure.transform_id_to_str
        )


class MeshProperty(DGeoFlowSubStructure):
    """meshproperties/meshproperties_x.json"""

    ContentVersion: Optional[str] = "2"
    Id: Optional[str] = None
    MeshProperties: Optional[List[PersistableMeshProperties]] = []

    if IS_PYDANTIC_V2:
        id_validator = field_validator("Id", mode="before")(
            DGeoFlowBaseModelStructure.transform_id_to_str
        )

    @classmethod
    def structure_name(cls) -> str:
        return "meshproperties"

    @classmethod
    def structure_group(cls) -> str:
        return "meshproperties"

    def add_meshproperty(
        self, layer_id: str, element_size: float, label: str
    ) -> PersistableMeshProperties:
        pmp = PersistableMeshProperties(
            LayerId=layer_id, Label=label, ElementSize=element_size
        )
        self.MeshProperties.append(pmp)
        return pmp

    def get_ids(self, exclude_soil_layer_id: Optional[int]) -> Set[str]:
        if exclude_soil_layer_id is not None:
            exclude_soil_layer_id = str(exclude_soil_layer_id)
        return {
            layer.LayerId
            for layer in self.MeshProperties
            if layer.LayerId != exclude_soil_layer_id
        }


class DGeoFlowStructure(BaseModelStructure):
    """Highest level DGeoFlow class that should be parsed to and serialized from.

    The List[] items (one for each scenario in the model) will be stored in a subfolder
    to multiple json files. Where the first (0) instance
    has no suffix, but the second one has (1 => _1) etc.

    also parses the outputs which are part of the json files
    """

    # input part
    soillayers: List[SoilLayerCollection] = [
        SoilLayerCollection(Id="14")
    ]  # soillayers/soillayers_x.json
    soils: SoilCollection = SoilCollection()  # soils.json
    soilvisualizations: SoilVisualisation = SoilVisualisation()  # soilvisualizations.json

    projectinfo: ProjectInfo = ProjectInfo()  # projectinfo.json
    geometries: List[Geometry] = [Geometry(Id="1")]  # geometries/geometry_x.json

    boundary_conditions: List[BoundaryConditionCollection] = [
        BoundaryConditionCollection(Id="15")
    ]  # boundaryconditions/boundaryconditions_x.json
    scenarios: List[Scenario] = [
        Scenario(
            Id="0",
            Label="Scenario 1",
            GeometryId="1",
            SoilLayersId="14",
            Stages=[
                PersistableStage(Label="Stage 1", BoundaryConditionCollectionId="15")
            ],
            Calculations=[
                PersistableCalculation(Label="Calculation 1", MeshPropertiesId="16")
            ],
        )
    ]  # scenarios/scenario_x.json
    mesh_properties: List[MeshProperty] = [
        MeshProperty(Id="16", MeshProperties=[])
    ]  # meshproperties/meshproperties_x.json

    # Output parts
    groundwater_flow_results: List[GroundwaterFlowResult] = []
    pipe_length_results: List[PipeLengthResult] = []
    critical_head_results: List[CriticalHeadResult] = []

    if IS_PYDANTIC_V2:
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    else:

        class Config:
            arbitrary_types_allowed = True
            validate_assignment = True

    def get_result_substructure(
        self, calculation_type: CalculationTypeEnum
    ) -> List[DGeoFlowResult]:
        result_types_mapping = {
            CalculationTypeEnum.GROUNDWATER_FLOW: self.groundwater_flow_results,
            CalculationTypeEnum.PIPE_LENGTH: self.pipe_length_results,
            CalculationTypeEnum.CRITICAL_HEAD: self.critical_head_results,
        }

        return result_types_mapping[calculation_type]

    if IS_PYDANTIC_V2:

        @model_validator(mode="after")
        def ensure_validity_foreign_keys(self):
            def list_has_id(values, id):
                for entry in values:
                    if entry.Id == id:
                        return True
                return False

            for _, scenario in enumerate(self.scenarios):
                for _, stage in enumerate(scenario.Stages):
                    if not list_has_id(
                        self.boundary_conditions, stage.BoundaryConditionCollectionId
                    ):
                        raise ValueError("BoundaryConditionCollectionIds not linked!")

                if not list_has_id(self.geometries, scenario.GeometryId):
                    raise ValueError("GeometryIds not linked!")
                if not list_has_id(self.soillayers, scenario.SoilLayersId):
                    raise ValueError("SoilLayersIds not linked!")

            return self

    else:
        # TODO: these are the original validators, but they are double?
        @root_validator(skip_on_failure=True, allow_reuse=True)
        def ensure_validity_foreign_keys(cls, values):
            stage_count = 0
            for i, scenario in enumerate(values.get("scenarios")):
                for _, stage in enumerate(scenario.Stages):
                    if (
                        stage.BoundaryConditionCollectionId
                        != values.get("boundary_conditions")[stage_count].Id
                    ):
                        raise ValueError("BoundaryConditionCollectionIds not linked!")
                    stage_count += 1

                if scenario.GeometryId != values.get("geometries")[i].Id:
                    raise ValueError("GeometryIds not linked!")
                if scenario.SoilLayersId != values.get("soillayers")[i].Id:
                    raise ValueError("SoilLayersIds not linked!")
            return values

        @root_validator(skip_on_failure=True, allow_reuse=True)
        def ensure_validity_foreign_keys(cls, values):
            def list_has_id(values, id):
                for entry in values:
                    if entry.Id == id:
                        return True
                return False

            for _, scenario in enumerate(values.get("scenarios")):
                for _, stage in enumerate(scenario.Stages):
                    if not list_has_id(
                        values.get("boundary_conditions"),
                        stage.BoundaryConditionCollectionId,
                    ):
                        raise ValueError("BoundaryConditionCollectionIds not linked!")

                if not list_has_id(values.get("geometries"), scenario.GeometryId):
                    raise ValueError("GeometryIds not linked!")
                if not list_has_id(values.get("soillayers"), scenario.SoilLayersId):
                    raise ValueError("SoilLayersIds not linked!")

            return values

    def add_default_scenario(
        self, label: str, notes: str, unique_start_id: Optional[int] = None
    ) -> Tuple[int, int]:
        """Add a new default (empty) scenario to DGeoFlow."""
        if unique_start_id is None:
            unique_start_id = self.get_unique_id()

        scenario_id = unique_start_id + 7
        self.soillayers += [SoilLayerCollection(Id=str(unique_start_id + 1))]
        self.mesh_properties += [MeshProperty(Id=str(unique_start_id + 2))]
        self.geometries += [Geometry(Id=str(unique_start_id + 3))]
        self.boundary_conditions += [
            BoundaryConditionCollection(Id=str(unique_start_id + 4))
        ]

        self.scenarios += [
            Scenario(
                Id=str(scenario_id),
                Label=label,
                Notes=notes,
                GeometryId=str(unique_start_id + 3),
                SoilLayersId=str(unique_start_id + 1),
                Calculations=[PersistableCalculation(Label="Calculation 1")],
                Stages=[
                    PersistableStage(
                        Label="Stage 1",
                        BoundaryConditionCollectionId=str(unique_start_id + 4),
                    )
                ],
            )
        ]

        return len(self.scenarios) - 1, scenario_id

    def add_default_stage(
        self,
        scenario_index: int,
        label: str,
        notes: str,
        unique_start_id: Optional[int] = None,
    ) -> int:
        """Add a new default (empty) stage to DStability."""
        if unique_start_id is None:
            unique_start_id = self.get_unique_id()

        self.boundary_conditions += [
            BoundaryConditionCollection(Id=str(unique_start_id + 1))
        ]

        new_stage = PersistableStage(
            Label=label,
            Notes=notes,
            BoundaryConditionCollectionId=str(unique_start_id + 1),
        )

        scenario = self.scenarios[scenario_index]

        if scenario.Stages is None:
            scenario.Stages = []

        scenario.Stages.append(new_stage)
        return len(scenario.Stages) - 1

    def add_default_calculation(
        self,
        scenario_index: int,
        label: str,
        notes: str,
    ) -> int:
        """Add a new default (empty) calculation to DStability."""

        new_calculation = PersistableCalculation(
            Label=label, Notes=notes, CalculationType=CalculationTypeEnum.GROUNDWATER_FLOW
        )

        scenario = self.scenarios[scenario_index]

        if scenario.Calculations is None:
            scenario.Calculations = []

        scenario.Calculations.append(new_calculation)
        return len(scenario.Calculations) - 1

    def get_unique_id(self) -> int:
        """Return unique id that can be used in DGeoFlow.
        Finds all existing ids, takes the max and does +1.
        """

        fk = ForeignKeys()
        classfields = fk.class_fields
        ids = []
        for instance in children(self):
            for field in classfields.get(instance.__class__.__name__, []):
                value = getattr(instance, field)
                if isinstance(value, (list, set, tuple)):
                    ids.extend(value)
                if isinstance(value, (int, float, str)):
                    ids.append(value)

        new_id = max({int(id) for id in ids if id is not None}) + 1
        return new_id

    def validator(self):
        return DGeoFlowValidator(self)

    def has_scenario(self, scenario_id: int) -> bool:
        try:
            self.scenarios[scenario_id]
            return True
        except IndexError:
            return False

    def has_result(self, scenario_index: int) -> bool:
        if self.has_scenario(scenario_index):
            result_id = self.scenarios[scenario_index].Calculations[0].ResultsId
            if result_id is None:
                return False
            else:
                return True
        return False

    def has_soil_layers(self, scenario_id: int) -> bool:
        if self.has_scenario(scenario_id):
            soil_layers_id = self.scenarios[scenario_id].SoilLayersId
            if soil_layers_id is None:
                return False
            else:
                return True
        return False

    def has_soil_layer(self, stage_id: int, soil_layer_id: int) -> bool:
        if self.has_soil_layers(stage_id):
            for layer in self.soillayers[stage_id].SoilLayers:
                if str(soil_layer_id) == layer.LayerId:
                    return True
            return False
        return False


class ForeignKeys(DGeoFlowBaseModelStructure):
    """A dataclass that store the connections between the
    various unique Ids used in DGeoFlow. These can be seen
    as (implicit) foreign keys.
    """

    mapping: Dict[str, Tuple[str, ...]] = {
        "PersistableSoil.Id": (
            "PersistableSoilVisualization.SoilId",
            "PersistableSoilLayer.SoilId",
        ),
        "PersistableLayer.Id": (
            "PersistableSoilLayer.LayerId",
            "PersistableMeshProperties.LayerId",
        ),
        "Geometry.Id": ("Scenario.GeometryId",),
        "SoilLayerCollection.Id": ("Scenario.SoilLayersId",),
        "BoundaryConditionCollection.Id": (
            "PersistableStage.BoundaryConditionCollectionId",
        ),
        "MeshProperty.Id": ("PersistableCalculation.MeshPropertiesId",),
        "Results.Id": ("PersistableCalculation.ResultsId",),
    }

    @property
    def class_fields(self) -> Dict[str, List[str]]:
        """Return a mapping in the form:
        classname: [fields]
        """
        id_keys = chain(*((k, *v) for k, v in self.mapping.items()))
        class_fields = defaultdict(list)
        for id_key in id_keys:
            classname, fieldname = id_key.split(".")
            class_fields[classname].append(fieldname)
        return class_fields
