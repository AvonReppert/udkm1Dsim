# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:17:35 2023

@author: matte
"""

import numpy as np
import udkm1Dsim as ud
u = ud.u


class Pt_111_1TM:
    def __init__(self):

        self.Pt = ud.Atom('Pt')

        self.prop = {}
        self.prop['crystal_struc'] = 'fcc'
        self.prop['c_axis'] = 2.2656*u.angstrom  # periodictable.com -- fcc geometry: 3.9242*np.sqrt(3)/3
        self.prop['a_axis'] = 2.5818*u.angstrom  # adjust density
        self.prop['b_axis'] = 2.5818*u.angstrom  # adjust density
        self.prop['density'] = 21450*u.kg/(u.m**3)  # periodictable.com
        self.prop['molar_mass'] = 195*u.g
        self.prop['deb_wal_fac'] = 0*u.m**2
        self.prop['elastic_c11'] = 375e9*u.kg/(u.m*u.s**2)  # 10.1016/0031-9163(66)90340-4: c11=347, c12=251, c44=76.5
        self.prop['elastic_c12'] = 242e9*u.kg/(u.m*u.s**2)  # 10.1016/0031-9163(66)90340-4: c11=347, c12=251, c44=76.5
        self.prop['elastic_c13'] = 232e9*u.kg/(u.m*u.s**2)  # 10.1016/0031-9163(66)90340-4: c11=347, c12=251, c44=76.5
        self.prop['elastic_c22'] = 375e9*u.kg/(u.m*u.s**2)  # 10.1016/0031-9163(66)90340-4: c11=347, c12=251, c44=76.5
        self.prop['elastic_c23'] = 232e9*u.kg/(u.m*u.s**2)  # 10.1016/0031-9163(66)90340-4: c11=347, c12=251, c44=76.5
        self.prop['elastic_c33'] = 386e9*u.kg/(u.m*u.s**2)  # 10.1016/0031-9163(66)90340-4: c11=347, c12=251, c44=76.5
        self.prop['sound_vel'] = 4.242*u.nm/u.ps  # calculated -- np.sqrt(c33/density)
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis'] = 8.9e-6/u.K  # 10.1103/PhysRev.61.74
        self.prop['exp_a_axis'] = 8.9e-6/u.K  # 10.1103/PhysRev.61.74
        self.prop['exp_b_axis'] = 8.9e-6/u.K  # 10.1103/PhysRev.61.74
        self.prop['lin_therm_exp'] = 19.58e-6  # calculated -- exp_c_axis*(1+2c23/c33)
        self.prop['Grun_c_axis'] = 2.6  # calculated -- lin_therm_exp*c33/heat_capacity
        self.prop['Grun_a_axis'] = 2.6  # calculated -- lin_therm_exp*c33/heat_capacity
        self.prop['Grun_b_axis'] = 2.6  # calculated -- lin_therm_exp*c33/heat_capacity
        self.prop['heat_capacity'] = 133*u.J/(u.kg * u.K)  # 10.1063/1.4959252
        self.prop['therm_cond'] = 71*u.W/(u.m * u.K)  # 10.1088/0022-3727/3/5/101
        self.prop['opt_pen_depth'] = 7.9*u.nm  # (800nm) below band gab
        self.prop['opt_ref_index'] = 0.5762 + 8.0776j  # (800nm) 10.1063/1.3243762
        self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        Pt = ud.UnitCell(name, 'Pt', caxis, **prop)
        Pt.add_atom(self.Pt, 0)
        return Pt


class Pt_111_2TM:
    def __init__(self):

        self.Pt = ud.Atom('Pt')

        self.prop = {}
        self.prop['crystal_struc'] = 'fcc'
        self.prop['c_axis'] = 2.2656*u.angstrom  # periodictable.com -- fcc geometry: 3.9242*np.sqrt(3)/3
        self.prop['a_axis'] = 2.5818*u.angstrom  # adjust density
        self.prop['b_axis'] = 2.5818*u.angstrom  # adjust density
        self.prop['density'] = 21450*u.kg/(u.m**3)  # periodictable.com
        self.prop['molar_mass'] = 195*u.g
        self.prop['deb_wal_fac'] = 0*u.m**2
        self.prop['elastic_c11'] = 375e9*u.kg/(u.m*u.s**2)  # 10.1016/0031-9163(66)90340-4: c11=347, c12=251, c44=76.5
        self.prop['elastic_c12'] = 242e9*u.kg/(u.m*u.s**2)  # 10.1016/0031-9163(66)90340-4: c11=347, c12=251, c44=76.5
        self.prop['elastic_c13'] = 232e9*u.kg/(u.m*u.s**2)  # 10.1016/0031-9163(66)90340-4: c11=347, c12=251, c44=76.5
        self.prop['elastic_c22'] = 375e9*u.kg/(u.m*u.s**2)  # 10.1016/0031-9163(66)90340-4: c11=347, c12=251, c44=76.5
        self.prop['elastic_c23'] = 232e9*u.kg/(u.m*u.s**2)  # 10.1016/0031-9163(66)90340-4: c11=347, c12=251, c44=76.5
        self.prop['elastic_c33'] = 386e9*u.kg/(u.m*u.s**2)  # 10.1016/0031-9163(66)90340-4: c11=347, c12=251, c44=76.5
        self.prop['sound_vel'] = 4.242*u.nm/u.ps  # calculated -- np.sqrt(c33/density)
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis'] = ['lambda T: 0.00084e-6*T', 8.9e-6/u.K]  # el: adjusted, ph: 10.1103/PhysRev.61.74
        self.prop['exp_a_axis'] = ['lambda T: 0.00084e-6*T', 8.9e-6/u.K]  # el: adjusted, ph: 10.1103/PhysRev.61.74
        self.prop['exp_b_axis'] = ['lambda T: 0.00084e-6*T', 8.9e-6/u.K]  # el: adjusted, ph: 10.1103/PhysRev.61.74
        # self.prop['lin_therm_exp'] = ['lambda T: 0.001848e-6*T', 19.58e-6]  # calculated -- exp_c_axis*(1+2c23/c33)
        self.prop['lin_therm_exp'] = ['lambda T: 0.002218e-6*T', 19.58e-6]  # calculated -- exp_c_axis*(1+2c23/c33)
        # self.prop['lin_therm_exp'] = ['lambda T: 0.002587e-6*T', 19.58e-6]  # calculated -- exp_c_axis*(1+2c23/c33)
        self.prop['Grun_c_axis'] = [1.0, 2.6]  # el: adjusted (2.3), ph: calculated -- lin_therm_exp*c33/heat_capacity
        self.prop['Grun_a_axis'] = [1.0, 2.6]  # el: adjusted (2.3), ph: calculated -- lin_therm_exp*c33/heat_capacity
        self.prop['Grun_b_axis'] = [1.0, 2.6]  # el: adjusted (2.3), ph: calculated -- lin_therm_exp*c33/heat_capacity
        # el -- 10.1016/S0301-0104(99)00330-4, ph -- 10.1063/1.4959252
        self.prop['heat_capacity'] = ['lambda T: 0.034*T', 133*u.J/(u.kg * u.K)]
        self.prop['therm_cond'] = ['lambda T: (66)*(T[0]/T[1])', 5*u.W/(u.m * u.K)]  # 10.1088/0022-3727/3/5/101
        self.prop['sub_system_coupling'] = [
            'lambda T:3.75e17*(T[1]-T[0])', 'lambda T:3.75e17*(T[0]-T[1])']  # 10.1063/4.0000120
        self.prop['opt_pen_depth'] = 7.9*u.nm  # (800nm) below band gab
        self.prop['opt_ref_index'] = 0.5762 + 8.0776j  # (800nm) 10.1063/1.3243762
        self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        Pt = ud.UnitCell(name, 'Pt', caxis, **prop)
        Pt.add_atom(self.Pt, 0)
        return Pt
