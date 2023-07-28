# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:44:23 2021

@author: matte
"""

import numpy as np
import udkm1Dsim as ud
u = ud.u


class Ta_111_1TM:
    def __init__(self):

        self.Ta = ud.Atom('Ta')

        self.prop = {}
        self.prop['crystal_struc'] = 'bcc'
        self.prop['c_axis'] = 2.859*u.angstrom  # periodictable.com -- bcc geometry: 3.301*sqrt(3)/2
        self.prop['a_axis'] = 3.547*u.angstrom  # adjust density
        self.prop['b_axis'] = 3.547*u.angstrom  # adjust density
        self.prop['molar_mass'] = 180.95*u.g
        self.prop['density'] = 16708*u.kg/(u.m**3)  # periodictable.com: 6.01e-25kg/35.97e-30m**3
        self.prop['deb_wal_fac'] = 0*u.m**2
        self.prop['elastic_c11'] = 291e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c12'] = 147e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c13'] = 137e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c22'] = 291e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c23'] = 137e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c33'] = 301e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['sound_vel'] = 4.24*u.nm/u.ps  # calculated -- np.sqrt(c33/density)
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis'] = 6.5e-6  # el: calculated, ph: Krischnan [S.120]
        self.prop['exp_a_axis'] = 6.5e-6  # el: calculated, ph: Krischnan [S.120]
        self.prop['exp_b_axis'] = 6.5e-6  # el: calculated, ph: Krischnan [S.120]
        self.prop['lin_therm_exp'] = 12.4e-6  # calculated: exp_c_axis*(1+2*c_13/c_33)
        self.prop['Grun_c_axis'] = 1.6  # Krischnan Book (1979) [S.70]
        self.prop['Grun_a_axis'] = 1.6  # Krischnan Book (1979) [S.70]
        self.prop['Grun_b_axis'] = 1.6  # Krischnan Book (1979) [S.70]
        self.prop['heat_capacity'] = 140*u.J/(u.kg * u.K)  # 10.1134/S0036029513090048
        self.prop['therm_cond'] = 57*u.W/(u.m * u.K)  # no reference available
        self.prop['opt_pen_depth'] = 10*u.nm  # (800nm) estimation
        self.prop['opt_ref_index'] = 0.99181 + 7.2934j  # (800nm) 10.1063/1.3243762
        self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        Ta = ud.UnitCell(name, 'Ta', caxis, **prop)
        Ta.add_atom(self.Ta, 0)
        Ta.add_atom(self.Ta, 0.5)
        return Ta


class Ta_111_2TM:
    def __init__(self):

        self.Ta = ud.Atom('Ta')

        self.prop = {}
        self.prop['crystal_struc'] = 'bcc'
        self.prop['c_axis'] = 2.859*u.angstrom  # periodictable.com -- bcc geometry: 3.301*sqrt(3)/2
        self.prop['a_axis'] = 3.547*u.angstrom  # adjust density
        self.prop['b_axis'] = 3.547*u.angstrom  # adjust density
        self.prop['molar_mass'] = 180.95*u.g
        self.prop['density'] = 16708*u.kg/(u.m**3)
        self.prop['deb_wal_fac'] = 0*u.m**2
        self.prop['elastic_c11'] = 291e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c12'] = 147e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c13'] = 137e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c22'] = 291e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c23'] = 137e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c33'] = 301e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['sound_vel'] = 4.24*u.nm/u.ps  # calculated -- np.sqrt(c33/density)
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis'] = ['lambda T: 0.00087e-6*T', 6.5e-6]  # el: calculated, ph: Krischnan [S.120]
        self.prop['exp_a_axis'] = ['lambda T: 0.00087e-6*T', 6.5e-6]  # el: calculated, ph: Krischnan [S.120]
        self.prop['exp_b_axis'] = ['lambda T: 0.00087e-6*T', 6.5e-6]  # el: calculated, ph: Krischnan [S.120]
        self.prop['lin_therm_exp'] = ['lambda T: 0.00166e-6*T', 12.4e-6]  # calculated: exp_c_axis*(1+2*c_13/c_33)
        self.prop['Grun_c_axis'] = [1.3, 1.6]  # Krischnan Book (1979) [S.70]
        self.prop['Grun_a_axis'] = [1.3, 1.6]  # Krischnan Book (1979) [S.70]
        self.prop['Grun_b_axis'] = [1.3, 1.6]  # Krischnan Book (1979) [S.70]
        self.prop['heat_capacity'] = ['lambda T: 0.023*T', 140*u.J/(u.kg * u.K)]  # 10.1134/S0036029513090048
        self.prop['therm_cond'] = ['lambda T: (52)*(T[0]/T[1])', 5*u.W/(u.m * u.K)]  # no reference available
        self.prop['sub_system_coupling'] = [
            'lambda T:1e17*(T[1]-T[0])', 'lambda T:1e17*(T[0]-T[1])']  # no reference available
        self.prop['opt_pen_depth'] = 10*u.nm  # (800nm) estimation
        self.prop['opt_ref_index'] = 0.99181 + 7.2934j  # (800nm) 10.1063/1.3243762
        self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        Ta = ud.UnitCell(name, 'Ta', caxis, **prop)
        Ta.add_atom(self.Ta, 0)
        Ta.add_atom(self.Ta, 0.5)
        return Ta


class Ta_111_3TM:
    def __init__(self):

        self.Ta = ud.Atom('Ta')

        self.prop = {}
        self.prop['crystal_struc'] = 'bcc'
        self.prop['c_axis'] = 2.859*u.angstrom  # periodictable.com -- bcc geometry: 3.301*sqrt(3)/2
        self.prop['a_axis'] = 3.547*u.angstrom  # adjust density
        self.prop['b_axis'] = 3.547*u.angstrom  # adjust density
        self.prop['molar_mass'] = 180.95*u.g
        self.prop['density'] = 16708*u.kg/(u.m**3)  # periodictable.com: 6.01e-25kg/35.97e-30m**3
        self.prop['deb_wal_fac'] = 0*u.m**2
        self.prop['elastic_c11'] = 291e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c12'] = 147e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c13'] = 137e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c22'] = 291e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c23'] = 137e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['elastic_c33'] = 301e9*u.kg/(u.m*u.s**2)  # 10.1103/PhysRev.130.1324 -- c11=261, c12=157, c44=82
        self.prop['sound_vel'] = 4.24*u.nm/u.ps  # calculated -- np.sqrt(c33/density)
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis'] = ['lambda T: 0.00087e-6*T', 6.5e-6, 0]  # el: calculated, ph: Krischnan [S.120]
        self.prop['exp_a_axis'] = ['lambda T: 0.00087e-6*T', 6.5e-6, 0]  # el: calculated, ph: Krischnan [S.120]
        self.prop['exp_b_axis'] = ['lambda T: 0.00087e-6*T', 6.5e-6, 0]  # el: calculated, ph: Krischnan [S.120]
        self.prop['lin_therm_exp'] = ['lambda T: 0.00166e-6*T', 12.4e-6, 0]  # calculated: exp_c_axis*(1+2*c_13/c_33)
        self.prop['Grun_c_axis'] = [1.3, 1.6, 0]  # Krischnan Book (1979) [S.70]
        self.prop['Grun_a_axis'] = [1.3, 1.6, 0]  # Krischnan Book (1979) [S.70]
        self.prop['Grun_b_axis'] = [1.3, 1.6, 0]  # Krischnan Book (1979) [S.70]
        self.prop['heat_capacity'] = ['lambda T: 0.023*T', 140*u.J/(u.kg * u.K), 1]  # 10.1134/S0036029513090048
        self.prop['therm_cond'] = ['lambda T: (52)*(T_0/T_1)', 5*u.W/(u.m * u.K), 0]  # no reference available
        self.prop['sub_system_coupling'] = ['1e17*(T_1-T_0)', '1e17*(T_0-T_1)', 0]  # no reference available
        self.prop['opt_pen_depth'] = 10*u.nm  # (800nm) estimation
        self.prop['opt_ref_index'] = 0.99181 + 7.2934j  # (800nm) 10.1063/1.3243762
        self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        Ta = ud.UnitCell(name, 'Ta', caxis, **prop)
        Ta.add_atom(self.Ta, 0)
        Ta.add_atom(self.Ta, 0.5)
        return Ta
