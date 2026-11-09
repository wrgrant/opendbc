"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import json
import math
from pathlib import Path

import numpy as np

from opendbc.car import DT_CTRL, structs
from opendbc.car.honda.values import CAR
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP

# Modified EPS firmware accepts larger requests than stock; the lookup maps them back to the EPS output range.
# Modified EPS cars not listed here use the stock linear STEER_MAX.
EPS_MOD_TORQUE_LOOKUP = {
  # stock request input values:     0x0000, 0x00DE, 0x014D, 0x01EF, 0x0290, 0x0377, 0x0454, 0x0610, 0x06EE
  # stock request output values:    0x0000, 0x0917, 0x0DC5, 0x1017, 0x119F, 0x140B, 0x1680, 0x1680, 0x1680
  # modified request output values: 0x0000, 0x0917, 0x0DC5, 0x1017, 0x119F, 0x140B, 0x1680, 0x2880, 0x3180
  # stock filter output values:     0x009F, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108
  # modified filter output values:  0x009F, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0400, 0x0480
  # note: max request allowed is 4096, but request is capped at 3840 in firmware, so modifications result in 2x max
  CAR.HONDA_CIVIC: ([0, 2560, 8000], [0, 2560, 3840]),
  CAR.HONDA_CIVIC_BOSCH: ([0, 2564, 8000], [0, 2564, 3840]),
  CAR.HONDA_CIVIC_BOSCH_DIESEL: ([0, 2564, 8000], [0, 2564, 3840]),
  CAR.HONDA_CIVIC_2022: ([0, 2564, 8000], [0, 2564, 3840]),
  # stock request input values:     0x0000, 0x00DB, 0x01BB, 0x0296, 0x0377, 0x0454, 0x0532, 0x0610, 0x067F
  # stock request output values:    0x0000, 0x0500, 0x0A15, 0x0E6D, 0x1100, 0x1200, 0x129A, 0x134D, 0x1400
  # modified request output values: 0x0000, 0x0500, 0x0A15, 0x0E6D, 0x1100, 0x1200, 0x1ACD, 0x239A, 0x2800
  CAR.HONDA_CRV_5G: ([0, 2560, 10000], [0, 2560, 3840]),
}


def get_eps_mod_torque_lookup(CP: structs.CarParams, CP_SP: structs.CarParamsSP) -> tuple[list[int], list[int]] | None:
  if not CP_SP.flags & HondaFlagsSP.EPS_MODIFIED:
    return None

  if CP.carFingerprint == CAR.HONDA_CLARITY:
    eps_fw = next((fw.fwVersion for fw in CP.carFw if fw.ecu == "eps"), b"")
    bp, v = ([0, 5760, 10240] if b"-" in eps_fw else [0, 5760, 15360]), [0, 2560, 3840]
  elif CP.carFingerprint in EPS_MOD_TORQUE_LOOKUP:
    bp, v = EPS_MOD_TORQUE_LOOKUP[CP.carFingerprint]
  else:
    return None

  # mirror around zero for signed requests
  return [-x for x in bp[:0:-1]] + bp, [-x for x in v[:0:-1]] + v


class EpsModCarController:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    self.eps_mod_lookup = get_eps_mod_torque_lookup(CP, CP_SP)
    self.odyssey_eps_map = None
    if CP.carFingerprint == CAR.HONDA_ODYSSEY and CP_SP.flags & HondaFlagsSP.EPS_MODIFIED:
      # Read once offroad at controller creation. Direct file avoids adding a
      # registered openpilot param outside this preparation task's write scope.
      try:
        config = Path('/data/params/d/OdysseyEpsMap').read_text()
      except OSError:
        config = None
      self.odyssey_eps_map = OdysseyEpsMap(config)
      self.odyssey_eps_last = 0.0

  def get_odyssey_apply_torque(self, limited_torque, speed):
    target = self.odyssey_eps_map.command(limited_torque, speed)
    # Bound CAN slew AFTER nonlinear/speed mapping, including configuration gain.
    step = self.params.STEER_MAX * self.params.STEER_DELTA_UP * DT_CTRL / self.odyssey_eps_map.scale
    self.odyssey_eps_last = float(np.clip(target, self.odyssey_eps_last - step, self.odyssey_eps_last + step))
    return int(self.odyssey_eps_last)

  def get_apply_torque(self, limited_torque: float) -> int:
    bp, v = self.eps_mod_lookup
    return int(np.interp(-limited_torque * bp[-1], bp, v))


# 39990-THR-A020 lane-keeping torque table: command (of 4096) to output. The output acts as a virtual torque on the
# wheel that goes through the normal assist curve (ODYSSEY.md, "Measured facts"), so a modified image is matched to
# stock by inverting its table, not by dividing the command by its scale.
ODYSSEY_TABLE_IN = np.arange(9) / 8
ODYSSEY_STOCK_OUT = np.array([0, 2019, 3230, 3980, 4345, 4577, 4721, 4864, 4864]) / 4864
ODYSSEY_FIRMWARE_OUT = {
  1.5: ODYSSEY_STOCK_OUT * 1.5,  # tools/odyssey_eps candidates: uniform scale
  2.0: ODYSSEY_STOCK_OUT * 2.0,
  2.5: np.array([0, 3968, 7936, 9950, 10863, 11443, 11803, 12160, 12160]) / 4864,  # community image
}


class OdysseyEpsMap:
  """Controller torque and speed to CAN command for an Odyssey with a modified EPS.

  The controller's torque goes through torque_bp/torque_v to a stock-equivalent command (above 1 asks for more than
  stock can give), then through the stock table to the virtual torque stock would produce, times the speed gain,
  capped at authority_ceiling (1 = stock's maximum). The command sent is the one whose modified-table output is that
  virtual torque. With the defaults a modified EPS gets the table output a stock one would.
  The firmware identity does not say which image is flashed: firmware_scale must match it. Without a valid config the
  largest scale is assumed, which sends the smallest commands.
  """
  def __init__(self, config=None):
    self.rows = None
    self.scale, self.ceiling = 2.5, 1.0
    self.speed_bp, self.speed_gain = [0, 40], [1, 1]
    self.torque_bp, self.torque_v = [0, 1], [0, 1]
    try:
      c = json.loads(config)
      scale = c['firmware_scale']
      ceiling = c.get('authority_ceiling', 1.0)
      sb, sg = c.get('speed_bp', [0, 40]), c.get('speed_gain', [1, 1])
      tb, tv = c.get('torque_bp', [0, 1]), c.get('torque_v', [0, 1])
      arrays = (sb, sg, tb, tv)
      valid = (type(scale) in (int, float) and scale in (1.5, 2.0, 2.5) and
               type(ceiling) in (int, float) and math.isfinite(ceiling) and 0 < ceiling <= 2 and
               all(isinstance(a, list) and 2 <= len(a) <= 32 and
                   all(type(x) in (int, float) and math.isfinite(x) for x in a) for a in arrays))
      if not valid:
        return
      if not (len(sb) == len(sg) and len(tb) == len(tv) and sb[0] == tb[0] == tv[0] == 0 and tb[-1] == 1 and
              0 < sb[-1] <= 100 and all(a < b for axis in (sb, tb) for a, b in zip(axis, axis[1:], strict=False)) and
              all(0 <= x <= 4 for x in sg + tv) and all(a <= b for a, b in zip(tv, tv[1:], strict=False))):
        return
      rows = c.get('torque_rows')
      if rows is not None:
        if not (isinstance(rows, list) and len(rows) == len(sb) and
                all(isinstance(row, list) and len(row) == len(tb) and row[0] == 0 and
                    all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 4 for x in row) and
                    all(a <= b for a, b in zip(row, row[1:], strict=False)) for row in rows)):
          return
      self.rows = rows
      self.scale, self.ceiling = scale, ceiling
      self.speed_bp, self.speed_gain, self.torque_bp, self.torque_v = sb, sg, tb, tv
    except (ValueError, TypeError, KeyError):
      pass

  def command(self, torque, speed):
    if not (math.isfinite(torque) and math.isfinite(speed)):
      return 0
    u, speed = min(abs(torque), 1), max(speed, 0)
    equivalent = np.interp(u, self.torque_bp, self.torque_v)
    if self.rows is not None:
      equivalent = np.interp(speed, self.speed_bp, [np.interp(u, self.torque_bp, row) for row in self.rows])
    virtual = np.interp(equivalent, ODYSSEY_TABLE_IN, ODYSSEY_STOCK_OUT) if equivalent <= 1 else equivalent
    virtual *= np.interp(speed, self.speed_bp, self.speed_gain)
    out = ODYSSEY_FIRMWARE_OUT[self.scale]
    virtual = min(virtual, self.ceiling, out[-1])
    command = np.interp(virtual, out[:-1], ODYSSEY_TABLE_IN[:-1])  # the last table point repeats the maximum
    return int(-math.copysign(command * 4096, torque))
