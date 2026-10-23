API Reference
===============

GTFSEnergyPredictor (Object-Oriented Interface)
------------------------------------------------

The primary interface for transit energy prediction.

.. autoclass:: routee.transit.GTFSEnergyPredictor
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

Network Routing
---------------

.. autofunction:: routee.transit.create_deadhead_shapes

.. autofunction:: routee.transit.gtfs_time_to_query_time

HVAC Energy
-----------

.. autofunction:: routee.transit.add_HVAC_energy

Depot and Agency Data (NTD)
---------------------------

.. autofunction:: routee.transit.load_ntd_facilities

.. autofunction:: routee.transit.match_agency_to_ntd

.. autoclass:: routee.transit.NTDAgencyMatch
    :members:
    :undoc-members:

TODS Export
-----------

.. autofunction:: routee.transit.write_tods_deadhead

Bundled Resources
-----------------

.. autofunction:: routee.transit.sample_inputs_path

.. autofunction:: routee.transit.ntd_path

GTFS Processing (Internal)
---------------------------

.. autofunction:: routee.transit.build_corridor_polygon

.. automodule:: routee.transit.gtfs_processing
    :members:
    :undoc-members:
    :show-inheritance:

Deadhead Trips (Internal)
--------------------------

.. automodule:: routee.transit.mid_block_deadhead
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: routee.transit.depot_deadhead
    :members:
    :undoc-members:
    :show-inheritance:

Transit Speed Model (Internal)
-------------------------------

.. automodule:: routee.transit.speed_model
    :members:
    :undoc-members:
    :show-inheritance:

Thermal Energy (Internal)
--------------------------

.. automodule:: routee.transit.thermal_energy
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: add_HVAC_energy

