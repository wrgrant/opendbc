"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import unittest

import numpy as np

from opendbc.testing import parameterized

from opendbc.car import DT_CTRL, gen_empty_fingerprint, structs
from opendbc.car.structs import CarParams
from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.values import CAR
from opendbc.car.honda.carcontroller import CruiseButtons
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP, HondaSafetyFlagsSP
from opendbc.sunnypilot.car.interfaces import setup_interfaces
from opendbc.sunnypilot.car.honda.eps_mod import EpsModCarController

CarFw = CarParams.CarFw


class TestHondaEpsMod(unittest.TestCase):

  @parameterized("car_name, fw", [(CAR.HONDA_CIVIC, b'39990-TBA,A030\x00\x00'), (CAR.HONDA_CIVIC, b'39990-TBA-A030\x00\x00'),
                                  (CAR.HONDA_CLARITY, b'39990-TRW-A020\x00\x00'), (CAR.HONDA_CLARITY, b'39990,TRW,A020\x00\x00')])
  def test_eps_mod_fingerprint(self, car_name, fw):
    fingerprint = gen_empty_fingerprint()
    car_fw = [CarFw(ecu="eps", fwVersion=fw)]

    CarInterface = interfaces[car_name]
    CP = CarInterface.get_params(car_name, fingerprint, car_fw, False, False, False)
    _ = CarInterface.get_params_sp(CP, car_name, fingerprint, car_fw, False, False, False)

    self.assertFalse(CP.dashcamOnly)

  @parameterized("fw, expected", [(b'39990-TBA,A030\x00\x00', [3840, 2560, -2898, -3840]),  # modified: nonlinear lookup
                                  (b'39990-TBA-A030\x00\x00', None)])  # stock: no lookup, upstream STEER_MAX applies
  def test_eps_mod_apply_torque(self, fw, expected):
    car_name = CAR.HONDA_CIVIC
    fingerprint = gen_empty_fingerprint()
    car_fw = [CarFw(ecu="eps", fwVersion=fw)]

    CarInterface = interfaces[car_name]
    CP = CarInterface.get_params(car_name, fingerprint, car_fw, False, False, False)
    CP.carFw = car_fw
    CP_SP = CarInterface.get_params_sp(CP, car_name, fingerprint, car_fw, False, False, False)

    CC = EpsModCarController(CP, CP_SP)
    if expected is None:
      self.assertIsNone(CC.eps_mod_lookup)
      return
    for limited_torque, exp in zip([-1.0, -0.32, 0.5, 1.2], expected, strict=True):
      self.assertEqual(CC.get_apply_torque(limited_torque), exp)


def _honda_ci(car_name, stock_long: bool):
  fingerprint = gen_empty_fingerprint()
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_params(car_name, fingerprint, [], False, False, False)
  CP_SP = CarInterface.get_params_sp(CP, car_name, fingerprint, [], False, False, False)
  setup_interfaces(CarInterface, CP, CP_SP, [{"HondaEnforceStockLongitudinal": "1" if stock_long else "0"}])
  return CarInterface(CP, CP_SP)


class TestHondaStockLongitudinal(unittest.TestCase):

  def test_nidec_stock_long_params(self):
    CI = _honda_ci(CAR.HONDA_ODYSSEY, True)
    self.assertFalse(CI.CP.openpilotLongitudinalControl)
    self.assertFalse(CI.CP.alphaLongitudinalAvailable)
    self.assertTrue(CI.CP.pcmCruise)
    self.assertTrue(CI.CP_SP.flags & HondaFlagsSP.STOCK_LONGITUDINAL)
    self.assertTrue(CI.CP_SP.safetyParam & HondaSafetyFlagsSP.STOCK_LONGITUDINAL)

  def test_nidec_default_is_openpilot_long(self):
    CI = _honda_ci(CAR.HONDA_ODYSSEY, False)
    self.assertTrue(CI.CP.openpilotLongitudinalControl)
    self.assertFalse(CI.CP_SP.flags & HondaFlagsSP.STOCK_LONGITUDINAL)
    self.assertFalse(CI.CP_SP.safetyParam & HondaSafetyFlagsSP.STOCK_LONGITUDINAL)

  def test_bosch_ignores_param(self):
    CI = _honda_ci(CAR.HONDA_CIVIC_BOSCH, True)
    self.assertFalse(CI.CP_SP.flags & HondaFlagsSP.STOCK_LONGITUDINAL)
    self.assertFalse(CI.CP_SP.safetyParam & HondaSafetyFlagsSP.STOCK_LONGITUDINAL)

  def _run(self, CI, enabled, cancel=False, frames=100):
    CC = structs.CarControl()
    CC.enabled = enabled
    CC.latActive = enabled
    CC.longActive = enabled
    CC.actuators.accel = -2.0  # hard brake request: must never turn into a brake command
    CC.cruiseControl.cancel = cancel
    CC_SP = structs.CarControlSP()
    sent = []
    now_nanos = 0
    for _ in range(frames):
      CI.update([])
      _, can_sends = CI.apply(CC.as_reader(), CC_SP, now_nanos)
      sent += can_sends
      now_nanos += int(DT_CTRL * 1e9)
    return sent

  def test_nidec_stock_long_sends_no_longitudinal_commands(self):
    CI = _honda_ci(CAR.HONDA_ODYSSEY, True)
    for enabled in (True, False):
      addrs = {addr for addr, _, _ in self._run(CI, enabled)}
      self.assertNotIn(0x1FA, addrs)  # BRAKE_COMMAND
      self.assertNotIn(0x30C, addrs)  # ACC_HUD (PCM gas and speed)
      self.assertNotIn(0x200, addrs)  # GAS_COMMAND
      self.assertNotIn(0xE5, addrs)   # BOSCH_SUPPLEMENTAL_1
      if enabled:
        self.assertIn(0xE4, addrs)    # still steers

  def test_nidec_stock_long_cancel_spams_buttons(self):
    CI = _honda_ci(CAR.HONDA_ODYSSEY, True)
    buttons = [dat for addr, dat, bus in self._run(CI, False, cancel=True, frames=10) if addr == 0x296 and bus == 0]
    self.assertEqual(len(buttons), 10)
    self.assertTrue(all(((dat[0] >> 5) & 0x7) == CruiseButtons.CANCEL for dat in buttons))

  def test_nidec_openpilot_long_still_sends_brake_and_acc_hud(self):
    CI = _honda_ci(CAR.HONDA_ODYSSEY, False)
    addrs = {addr for addr, _, _ in self._run(CI, True)}
    self.assertIn(0x1FA, addrs)
    self.assertIn(0x30C, addrs)


class TestOdysseyMap(unittest.TestCase):
  @staticmethod
  def table_out(scale, command):
    """Modified-table output for a CAN command, relative to stock's maximum."""
    from opendbc.sunnypilot.car.honda.eps_mod import ODYSSEY_FIRMWARE_OUT, ODYSSEY_TABLE_IN
    return float(np.interp(abs(command) / 4096, ODYSSEY_TABLE_IN, ODYSSEY_FIRMWARE_OUT[scale]))

  def test_default_and_invalid(self):
    from opendbc.sunnypilot.car.honda.eps_mod import OdysseyEpsMap
    for config in (None, '', '{}', 'null', '{"firmware_scale":NaN}'):
      m = OdysseyEpsMap(config)
      self.assertEqual(m.scale, 2.5)  # the largest scale: the smallest commands
      self.assertEqual(m.command(1, 10), -627)
      self.assertGreater(m.command(-0.5, 0), 0)

  def test_default_map_gives_the_stock_table_output(self):
    import json
    from opendbc.sunnypilot.car.honda.eps_mod import ODYSSEY_STOCK_OUT, ODYSSEY_TABLE_IN, OdysseyEpsMap
    for scale in (1.5, 2.0, 2.5):
      m = OdysseyEpsMap(json.dumps(dict(firmware_scale=scale)))
      for u in np.linspace(0.05, 1, 20):
        stock = float(np.interp(u, ODYSSEY_TABLE_IN, ODYSSEY_STOCK_OUT))
        self.assertAlmostEqual(self.table_out(scale, m.command(u, 20)), stock, delta=0.004)
        self.assertEqual(m.command(u, 20), -m.command(-u, 20))
      self.assertLess(abs(m.command(1, 20)), 4096 / scale + 1)

  def test_ceiling_speed_gain_and_extra_authority(self):
    import json
    from opendbc.sunnypilot.car.honda.eps_mod import OdysseyEpsMap
    base = dict(firmware_scale=1.5, speed_bp=[0, 30], speed_gain=[2, 1], torque_bp=[0, .8, 1], torque_v=[0, .8, 1.5])
    capped = OdysseyEpsMap(json.dumps(dict(base, authority_ceiling=1.0)))
    self.assertAlmostEqual(self.table_out(1.5, capped.command(1, 30)), 1.0, delta=0.004)  # the knee asks for 1.5: capped
    self.assertAlmostEqual(self.table_out(1.5, capped.command(1, 0)), 1.0, delta=0.004)  # the speed gain asks for 2: capped
    raised = OdysseyEpsMap(json.dumps(dict(base, authority_ceiling=1.3)))
    self.assertAlmostEqual(self.table_out(1.5, raised.command(1, 30)), 1.3, delta=0.004)
    self.assertAlmostEqual(self.table_out(1.5, raised.command(.4, 30)), self.table_out(1.5, capped.command(.4, 30)), delta=1e-6)
    full = OdysseyEpsMap(json.dumps(dict(base, authority_ceiling=2.0)))
    self.assertAlmostEqual(self.table_out(1.5, full.command(1, 0)), 1.5, delta=0.004)  # never beyond what the image holds
    self.assertEqual(capped.command(float('nan'), 0), 0)

  def test_actual_odyssey_can_and_fingerprint(self):
    from unittest.mock import patch
    import math
    from opendbc.sunnypilot.car.honda.fingerprints_ext import FW_VERSIONS_EXT, Ecu
    self.assertIn(b'39990-THR,A020\0\0', FW_VERSIONS_EXT[CAR.HONDA_ODYSSEY][(Ecu.eps, 0x18da30f1, None)])
    for modified in (False, True):
      fw = b'39990-THR,A020\0\0' if modified else b'39990-THR-A020\0\0'
      CIClass = interfaces[CAR.HONDA_ODYSSEY]
      fingerprint = gen_empty_fingerprint()
      car_fw = [CarFw(ecu='eps', fwVersion=fw)]
      cp = CIClass.get_params(CAR.HONDA_ODYSSEY, fingerprint, car_fw, False, False, False)
      cp.carFw = car_fw
      sp = CIClass.get_params_sp(cp, CAR.HONDA_ODYSSEY, fingerprint, car_fw, False, False, False)
      self.assertEqual(bool(sp.flags & HondaFlagsSP.EPS_MODIFIED), modified)
      with patch('opendbc.sunnypilot.car.honda.eps_mod.Path.read_text', side_effect=FileNotFoundError):
        ci = CIClass(cp, sp)
      cc = structs.CarControl(latActive=True)
      cc.actuators.torque = 1
      for frame in range(40):
        ci.update([])
        _, sends = ci.apply(cc.as_reader(), structs.CarControlSP(), frame*10000000)
        dat = next(dat for addr, dat, _ in sends if addr == 0xE4)
        actual = int.from_bytes(dat[:2], 'big', signed=True)
        limited = min((frame+1)*.03, 1)
        if not modified:
          self.assertLessEqual(abs(actual - math.trunc(-limited * 4096)), 1)  # float accumulation/truncation
        self.assertLessEqual(abs(actual), (frame + 1) * 4096 * 0.03 / (2.5 if modified else 1) + 1)  # CAN slew bound
        if frame == 39:
          self.assertEqual(actual, -627 if modified else -4096)
