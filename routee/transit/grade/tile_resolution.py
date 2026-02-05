from __future__ import annotations

from enum import Enum


class TileResolution(Enum):
    ONE_THIRD_ARC_SECOND = 13

    # TODO: gradeit currently only support one third arc second resolution;
    #   Update this when gradeit supports one arc second
    # ONE_ARC_SECOND = 1

    @classmethod
    def from_string(cls, string: str) -> TileResolution:
        if string.lower() in ["1/3", "one third"]:
            return TileResolution.ONE_THIRD_ARC_SECOND
        # elif string.lower() in ["1", "one"]:
        #     return TileResolution.ONE_ARC_SECOND
        else:
            raise ValueError(
                f"invalid string {string} for tile resolution. Must be one of: 1, one, 1/3, one third"
            )

    @classmethod
    def from_int(cls, int: int) -> TileResolution:
        if int == 13:
            return TileResolution.ONE_THIRD_ARC_SECOND
        # elif int == 1:
        #     return TileResolution.ONE_ARC_SECOND
        else:
            raise ValueError(
                f"invalid int {int} for tile resolution. Must be one of: 1, 13"
            )
