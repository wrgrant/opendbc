"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from enum import IntFlag


class HondaFlagsSP(IntFlag):
  NIDEC_HYBRID = 1
  EPS_MODIFIED = 2
  HYBRID_ALT_BRAKEHOLD = 4
  STOCK_LONGITUDINAL = 16  # same bit as mvl-boston's fork, so logs and params read the same


class HondaSafetyFlagsSP:
  NIDEC_HYBRID = 1
  GAS_INTERCEPTOR = 2
  STOCK_LONGITUDINAL = 4
