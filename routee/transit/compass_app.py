from nrel.routee.compass import CompassApp
from routee.transit.routee_transit_py import TransitCompassAppPy


class TransitCompassApp(CompassApp):
    """
    A custom CompassApp for RouteE-Transit that uses the TransitCompassAppPy
    constructor from the Rust extension.
    """

    @classmethod
    def get_constructor(cls) -> type[TransitCompassAppPy]:
        """
        Return the underlying constructor for the application.
        """
        return TransitCompassAppPy  # type: ignore[no-any-return]
