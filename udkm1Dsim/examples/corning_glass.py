# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:44:23 2021

@author: matte
"""

import numpy as np
import udkm1Dsim as ud
u = ud.u


class corning_glass_1TM:
    def __init__(self):

        self.He = ud.Atom('He')

        self.prop = {}
        self.prop['crystal_struc'] = 'polycrystalline'
        self.prop['c_axis'] = 2.859*u.angstrom  # chosed to avoid overlap in reciprocal space
        self.prop['a_axis'] = 0.956*u.angstrom  # adjust density
        self.prop['b_axis'] = 0.956*u.angstrom  # adjust density
        self.prop['molar_mass'] = 4*u.g  # He to supress Bragg peak in UXRD simulation
        self.prop['density'] = 2541*u.kg/(u.m**3)  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['deb_wal_fac'] = 0*u.m**2
        self.prop['elastic_c11'] = 83e9*u.kg/(u.m*u.s**2)  # calculated -- sound_vel^2*density
        self.prop['elastic_c12'] = 20e9*u.kg/(u.m*u.s**2)  # guess
        self.prop['elastic_c13'] = 20e9*u.kg/(u.m*u.s**2)  # guess
        self.prop['elastic_c22'] = 83e9*u.kg/(u.m*u.s**2)  # calculated -- sound_vel^2*density
        self.prop['elastic_c23'] = 20e9*u.kg/(u.m*u.s**2)  # guess
        self.prop['elastic_c33'] = 83e9*u.kg/(u.m*u.s**2)  # calculated -- sound_vel^2*density
        self.prop['sound_vel'] = 5.7*u.nm/u.ps  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis'] = 4.2e-6/u.K  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['exp_a_axis'] = 4.2e-6/u.K  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['exp_b_axis'] = 4.2e-6/u.K  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['lin_therm_exp'] = 6.2e-6  # calculated -- exp_c_axis*(1+2*c_13/c_33)
        self.prop['Grun_c_axis'] = 0.3  # calculated -- exp_c_axis*(c_13+c_23+c_33)/(heat_capacity*density)
        self.prop['Grun_a_axis'] = 0.3  # calculated -- exp_c_axis*(c_13+c_23+c_33)/(heat_capacity*density)
        self.prop['Grun_b_axis'] = 0.3  # calculated -- exp_c_axis*(c_13+c_23+c_33)/(heat_capacity*density)
        self.prop['heat_capacity'] = 707*u.J/(u.kg * u.K)  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['therm_cond'] = 1*u.W/(u.m * u.K)  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['opt_pen_depth'] = np.inf*u.nm  # (800nm) below band gab
        self.prop['opt_ref_index'] = 1.5  # (800nm, estimated) Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        glass = ud.UnitCell(name, 'corning_glass', caxis, **prop)
        glass.add_atom(self.He, 0)
        return glass


class corning_glass_2TM:
    def __init__(self):

        self.He = ud.Atom('He')

        self.prop = {}
        self.prop['crystal_struc'] = 'polycrystalline'
        self.prop['c_axis'] = 2.859*u.angstrom  # chosed to avoid overlap in reciprocal space
        self.prop['a_axis'] = 0.956*u.angstrom  # adjust density
        self.prop['b_axis'] = 0.956*u.angstrom  # adjust density
        self.prop['molar_mass'] = 4*u.g  # He to supress Bragg peak in UXRD simulation
        self.prop['density'] = 2541*u.kg/(u.m**3)  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['deb_wal_fac'] = 0*u.m**2
        self.prop['elastic_c11'] = 83e9*u.kg/(u.m*u.s**2)  # calculated -- sound_vel^2*density
        self.prop['elastic_c12'] = 20e9*u.kg/(u.m*u.s**2)  # guess
        self.prop['elastic_c13'] = 20e9*u.kg/(u.m*u.s**2)  # guess
        self.prop['elastic_c22'] = 83e9*u.kg/(u.m*u.s**2)  # calculated -- sound_vel^2*density
        self.prop['elastic_c23'] = 20e9*u.kg/(u.m*u.s**2)  # guess
        self.prop['elastic_c33'] = 83e9*u.kg/(u.m*u.s**2)  # calculated -- sound_vel^2*density
        self.prop['sound_vel'] = 5.7*u.nm/u.ps  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis'] = [0, 4.2e-6]  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['exp_a_axis'] = [0, 4.2e-6]  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['exp_b_axis'] = [0, 4.2e-6]  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['lin_therm_exp'] = [0, 6.2e-6]  # Calculation: exp_c_axis*(1+2*c_13/c_33)
        self.prop['Grun_c_axis'] = [0, 0.3]  # calculated -- exp_c_axis*(c_13+c_23+c_33)/(heat_capacity*density)
        self.prop['Grun_a_axis'] = [0, 0.3]  # calculated -- exp_c_axis*(c_13+c_23+c_33)/(heat_capacity*density)
        self.prop['Grun_b_axis'] = [0, 0.3]  # calculated -- exp_c_axis*(c_13+c_23+c_33)/(heat_capacity*density)
        self.prop['heat_capacity'] = [0.01, 707*u.J/(u.kg * u.K)]
        # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['therm_cond'] = [0, 2]  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['sub_system_coupling'] = ['lambda T:1e12*(T[1]-T[0])', 'lambda T:1e12*(T[0]-T[1])']
        self.prop['opt_pen_depth'] = np.inf*u.nm  # (800nm) below band gab
        self.prop['opt_ref_index'] = 1.5  # (800nm, estimated) Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        glass = ud.UnitCell(name, 'corning_glass', caxis, **prop)
        glass.add_atom(self.He, 0)
        return glass


class corning_glass_3TM:
    def __init__(self):

        self.He = ud.Atom('He')

        self.prop = {}
        self.prop['crystal_struc'] = 'polycrystalline'
        self.prop['c_axis'] = 2.859*u.angstrom  # chosed to avoid overlap in reciprocal space
        self.prop['a_axis'] = 0.956*u.angstrom  # adjust density
        self.prop['b_axis'] = 0.956*u.angstrom  # adjust density
        self.prop['molar_mass'] = 4*u.g  # He to supress Bragg peak in UXRD simulation
        self.prop['density'] = 2541*u.kg/(u.m**3)  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['deb_wal_fac'] = 0*u.m**2
        self.prop['elastic_c11'] = 83e9*u.kg/(u.m*u.s**2)  # calculated -- sound_vel^2*density
        self.prop['elastic_c12'] = 20e9*u.kg/(u.m*u.s**2)  # guess
        self.prop['elastic_c13'] = 20e9*u.kg/(u.m*u.s**2)  # guess
        self.prop['elastic_c22'] = 83e9*u.kg/(u.m*u.s**2)  # calculated -- sound_vel^2*density
        self.prop['elastic_c23'] = 20e9*u.kg/(u.m*u.s**2)  # guess
        self.prop['elastic_c33'] = 83e9*u.kg/(u.m*u.s**2)  # calculated -- sound_vel^2*density
        self.prop['sound_vel'] = 5.7*u.nm/u.ps  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis'] = [0, 4.2e-6]  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['exp_a_axis'] = [0, 4.2e-6]  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['exp_b_axis'] = [0, 4.2e-6]  # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['lin_therm_exp'] = [0, 6.2e-6, 0]  # calculated -- exp_c_axis*(1+2*c_13/c_33)
        self.prop['Grun_c_axis'] = [0, 0.3]  # calculated -- exp_c_axis*(c_13+c_23+c_33)/(heat_capacity*density)
        self.prop['Grun_a_axis'] = [0, 0.3]  # calculated -- exp_c_axis*(c_13+c_23+c_33)/(heat_capacity*density)
        self.prop['Grun_b_axis'] = [0, 0.3]  # calculated -- exp_c_axis*(c_13+c_23+c_33)/(heat_capacity*density)
        self.prop['heat_capacity'] = [0.01, 707*u.J/(u.kg * u.K), 1]
        # Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['therm_cond'] = [0, 1, 0]  # (W/m*K) Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['sub_system_coupling'] = ['1e12*(T_1-T_0)', '1e12*(T_0-T_1)', 0]
        self.prop['opt_pen_depth'] = np.inf*u.nm  # (800nm) below band gab
        self.prop['opt_ref_index'] = 1.5  # (800nm, estimated) Corning Incorporated, Corning 1737 AMLCD Glass, 2002
        self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        glass = ud.UnitCell(name, 'corning_glass', caxis, **prop)
        glass.add_atom(self.He, 0)
        return glass
