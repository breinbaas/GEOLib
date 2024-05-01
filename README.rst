GEOLib
=============================

GEOLib is a Python package to generate, execute and parse several D-Serie and D-GEO Suite numerical models.

What's different with this fork
-------------------------------

This fork is maintained by Rob van Putten aka Breinbaas aka LeveeLogic and contains extra functionality
mostly for DStability which makes it easier to access or use geolib for levee assessments. The following
extras are implemented;

* Automatic upgrade of older stix files

If you parse an old stix file the migration console will be called to upgrade the file. Note that you
will need to add the DSTABILITY_MIGRATION_CONSOLE_PATH keyword to the geolib.env file. See the geolib.env
for an example.

* Use string or Path as the parse parameter

It is possible to call the parse method of the DStabilityModel with a string as parameters which 
saves you the code to convert the argument to a Path.

* Get the geometry limits

The xmin, xmax, zmin, zmax properties are added to the DStabilityModel which refer to the geometry
limits of the selected scenario / stage

.. code-block:: python

    dm = DStabilityModel()
    ...
    print(dm.zmax, dm.zmin, dm.xmax, dm.xmin)

* Get a dictionary with layer / soil information

It can be pretty useful to know which layer contains which soil. That's now possible using the layer_soil_dict property.
This dictionary contains the layerId as key and the soil as a value.

.. code-block:: python

    dm = DStabilityModel()
    ...
    print(dm.layer_soil_dict)

* Identify a layer at a given point

It is possible to get a layer from a given point.

.. code-block:: python

    dm = DStabilityModel()
    ...
    dm.layer_at(x=0.0, z=-10.0)

* Get layer by name or label

.. code-block:: python

    dm = DStabilityModel()
    ...
    dm.layer_by_id('51')
    dm.layer_by_label('L 1')

* Get soil layer intersection at a given x coordinate

Use the next function to get a list of layers from top to bottom that intersect at a given x coordinate

.. code-block:: python

    dm = DStabilityModel()
    ...
    dm.layer_intersections_at(0.0)

* Get the characteristic points

The characteristic points which can be found in the waternet creator settings are a bit hidden.. but not anymore!

.. code-block:: python

    dm = DStabilityModel()
    ...
    dm.get_characteristic_point(CharacteristicPointEnum.EMBANKEMENT_TOE_WATER_SIDE)

    # possible values for the CharacteristicPointEnum are;
    #
    # NONE
    # EMBANKEMENT_TOE_WATER_SIDE
    # EMBANKEMENT_TOP_WATER_SIDE
    # EMBANKEMENT_TOP_LAND_SIDE
    # SHOULDER_BASE_LAND_SIDE
    # EMBANKEMENT_TOE_LAND_SIDE
    # DITCH_EMBANKEMENT_SIDE
    # DITCH_BOTTOM_EMBANKEMENT_SIDE
    # DITCH_BOTTOM_LAND_SIDE
    # DITCH_LAND_SIDE


* Get the waternet creator settings

.. code-block:: python

    dm = DStabilityModel()
    ...
    dm._get_waternetcreator_settings() 

* Get the surface of the geometry

You can easily get the points that define the surface of the geometry or the points that define the ditch as a list of x,z tuples using the following code;

.. code-block:: python

    dm = DStabilityModel()
    ...
    dm.surface 
    dm.ditch_points

* Get the z coordinate(s) at a given x coordinate

Use the next function to get the intersection with the soillayers at the given x coordinate. You can choose to only return the highest intersection or all intersections from top to bottom

.. code-block:: python

    dm = DStabilityModel()
    ...
    dm.z_at(x=0, highest_only=False)

* Run the waternet creator from code

It is possible to run the waternet creator from code. You can either use parameters or the already defined waternet creator settings if the are
available in the model. 

**NOTE** this code is under development and testing is needed, use at your own risk!

Here is an example where the input of the waternet creator settings in the file is used to find in the required parameters;

.. code-block:: python

    dm = DStabilityModel()
    ...
    dm.generate_waternet(river_level_mhw=2.0, river_level_ghw=0.0, polder_level=-1.0)

And here is an example where the user defines the necessary parameters in the function (note that if 
parameters are missing the code tries to get them from the available waternet creator settings;

.. code-block:: python

    from geolib.internal import EmbankmentSoilScenarioEnum

    dm = DStabilityModel()
    ...
    dm.generate_waternet(
        river_level_mhw = 2.0,
        river_level_ghw = 0.0,
        polder_level = -1.0,
        x_embankment_toe_land_side = 10.0,
        x_embankment_top_land_side = 15.0,
        x_embankment_toe_water_side = 25.0,
        material_layout = EmbankmentSoilScenarioEnum["CLAY_EMBANKMENT_ON_CLAY]",
        aquifer_label = 'L 1',
        aquifer_inside_aquitard_label = 'L 4',
        intrusion_length = 3.0,
        hydraulic_head_pl2_inward = -0.5,
        hydraulic_head_pl2_outward = -0.5,
        inward_leakage_length_pl3 = 650,
        outward_leakage_length_pl3 = 10,
        inward_leakage_length_pl4 = 1000,
        outward_leakage_length_pl4 = 30,
    )

**NOTE** adjust for uplift is not yet implemented

Installation
------------

Install GEOLib with:

.. code-block:: bash

    $ pip install d-geolib

Configure your environment using the instructions on our `Setup <https://deltares.github.io/GEOLib/latest/user/setup.html>`_ page.
You may get the console executables from the Deltares download portal, or in the case of the D-GEO Suite, you may copy the contents of the installation 'bin' directory to your console folder.

Running the source code
-----------------------

If you want to make changes to GEOLib you can run the source code from GitHub directly on your local machine, 
please follow the instructions below on how to set up your development environment using pip or poetry.

You do not need to follow these instructions if you want to use the GEOLib package in your project.

Requirements
------------

To install the required dependencies to run GEOLib code, run:

.. code-block:: bash

    $ pip install -r requirements.txt

Or, when having poetry installed (you should):

.. code-block:: bash

    $ poetry install


Testing & Development
---------------------

Make sure to have the server dependencies installed: 

.. code-block:: bash

    $ poetry install -E server

In order to run the testcode, from the root of the repository, run:

.. code-block:: bash

    $ pytest

or, in case of using Poetry

.. code-block:: bash

    $ poetry run pytest

Running flake8, mypy is also recommended. For mypy use:

.. code-block:: bash

    $ mypy --config-file pyproject.toml geolib

Running standard linters is advised:

.. code-block:: bash

    $ poetry run isort .
    $ poetry run black .


Documentation
-------------

In order to run the documentation, from the root of the repository, run:

.. code-block:: bash

    $ cd docs
    $ sphinx-build . build -b html -c .


The documentation is now in the `build` subfolder, where you can open 
the `index.html` in your browser.

Build wheel
-----------

To build a distributable wheel package, run:

.. code-block:: bash

    $ poetry build

The distributable packages are now built in the `dist` subfolder.

Update requirements.txt
-----------------------

The requirements.txt file is generated by poetry based on the pyproject.toml and poetry.lock files. In order to update/regenerate this file, run:

.. code-block:: bash

    $ poetry install
    $ poetry export -E server -f requirements.txt --output requirements.txt --without-hashes
    $ poetry export -E server -f requirements.txt --output requirements-dev.txt --with dev --without-hashes

Code linter
-----------------------

In order to run code cleanup/linter use the following commands:

.. code-block:: bash

    $ poetry run isort .
    $ poetry run black .