# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:44:23 2021

@author: matte
"""

import numpy as np
import udkm1Dsim as ud
u = ud.u


class Ni_111_1TM:
    def __init__(self):

        self.Ni = ud.Atom('Ni')

        self.prop = {}
        self.prop['crystal_struc'] = 'fcc'
        self.prop['c_axis'] = 2.035*u.angstrom  # periodictable.com -- fcc geometry: 3.524*np.sqrt(3)/3
        self.prop['a_axis'] = 2.3185*u.angstrom  # adjust density
        self.prop['b_axis'] = 2.3185*u.angstrom  # adjust density
        self.prop['molar_mass'] = 58.69*u.g
        self.prop['density'] = 8912*u.kg/(u.m**3)
        self.prop['deb_wal_fac'] = 0*u.m**2
        self.prop['elastic_c11'] = 327e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c12'] = 128e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c13'] = 103e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c22'] = 327e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c23'] = 103e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c33'] = 351e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['sound_vel'] = 6.3*u.nm/u.ps  # calculated -- np.sqrt(c33/density)
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis'] = 12.7e-6  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        self.prop['exp_a_axis'] = 12.7e-6  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        self.prop['exp_b_axis'] = 12.7e-6  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        self.prop['lin_therm_exp'] = 20.2e-6  # calculated: exp_c_axis*(1+2*c_13/c_33)
        self.prop['Grun_c_axis'] = 1.8  # 10.1063/1.2902170
        self.prop['Grun_a_axis'] = 1.8  # 10.1063/1.2902170
        self.prop['Grun_b_axis'] = 1.8  # 10.1063/1.2902170
        self.prop['heat_capacity'] = 442*u.J/(u.kg * u.K)  # ph: 10.1016/0022-3697(81)90174-8
        self.prop['therm_cond'] = 91*u.W/(u.m * u.K)  # 10.1016/S0301-0104(99)00330-4
        self.prop['opt_pen_depth'] = 15*u.nm  # (800nm) estimation
        self.prop['opt_ref_index'] = 2.3223 + 8.8820j  # (800nm, thick film) Werner, J. Phys. Chem. Ref. Data 38 (2009)
        self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        Ni = ud.UnitCell(name, 'Ni', caxis, **prop)
        Ni.add_atom(self.Ni, 0)
        return Ni



class Ni_111_2TM:
    def __init__(self):

        self.Ni = ud.Atom('Ni')

        self.prop = {}
        self.prop['crystal_struc'] = 'fcc'
        self.prop['c_axis'] = 2.035*u.angstrom  # periodictable.com -- fcc geometry: 3.524*np.sqrt(3)/3
        self.prop['a_axis'] = 2.3185*u.angstrom  # adjust density
        self.prop['b_axis'] = 2.3185*u.angstrom  # adjust density
        self.prop['molar_mass'] = 58.69*u.g
        self.prop['density'] = 8912*u.kg/(u.m**3)
        self.prop['deb_wal_fac'] = 0*u.m**2
        self.prop['elastic_c11'] = 327e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c12'] = 128e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c13'] = 103e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c22'] = 327e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c23'] = 103e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c33'] = 351e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['sound_vel'] = 6.3*u.nm/u.ps  # calculated -- np.sqrt(c33/density)
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis'] = ['lambda T: 0.0026e-6*T', 12.7e-6]  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        self.prop['exp_a_axis'] = ['lambda T: 0.0026e-6*T', 12.7e-6]  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        self.prop['exp_b_axis'] = ['lambda T: 0.0026e-6*T', 12.7e-6]  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        self.prop['lin_therm_exp'] = ['lambda T: 0.0041e-6*T', 20.2e-6]  # calculated: exp_c_axis*(1+2*c_13/c_33)
        self.prop['Grun_c_axis'] = [1.35, 1.8]  # 10.1063/1.2902170
        self.prop['Grun_a_axis'] = [1.35, 1.8]  # 10.1063/1.2902170
        self.prop['Grun_b_axis'] = [1.35, 1.8]  # 10.1063/1.2902170
        self.prop['heat_capacity'] = ['lambda T: 0.12*T', 442*u.J/(u.kg * u.K)]
        # 0.08*T (Zahn 2021, 10.1103/PhysRevResearch.3.023032)
        # el: 10.1016/S0301-0104(99)00330-4, ph: 10.1016/0022-3697(81)90174-8
        self.prop['sub_system_coupling'] = [
            'lambda T:3.6e17*(T[1]-T[0])', 'lambda T:3.6e17*(T[0]-T[1])']  # 10.1103/PhysRevB.77.075133
        # self.prop['sub_system_coupling'] = ['17e17*(T_1-T_0)', '17e17*(T_0-T_1)']  # 10.1103/PhysRevResearch.3.023032
        self.prop['therm_cond'] = ['lambda T: (81.4)*(T[0]/T[1])', 9.6]  # 10.1016/S0301-0104(99)00330-4
        self.prop['opt_pen_depth'] = 15*u.nm  # (800nm) estimation
        self.prop['opt_ref_index'] = 2.3223 + 8.8820j  # (800nm, thick film) Werner, J. Phys. Chem. Ref. Data 38 (2009)
        self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        Ni = ud.UnitCell(name, 'Ni', caxis, **prop)
        Ni.add_atom(self.Ni, 0)
        return Ni

class Ni_111_LLB:
    def __init__(self):

        self.Ni = ud.Atom('Ni', mag_amplitude=1, mag_gamma=90*u.deg, mag_phi=0*u.deg)

        self.prop = {}
        self.prop['crystal_struc'] = 'fcc'
        self.prop['c_axis'] = 2.035*u.angstrom  # periodictable.com -- fcc geometry: 3.524*np.sqrt(3)/3
        self.prop['a_axis'] = 2.3185*u.angstrom  # adjust density
        self.prop['b_axis'] = 2.3185*u.angstrom  # adjust density
        self.prop['molar_mass'] = 58.69*u.g
        self.prop['density'] = 8912*u.kg/(u.m**3)
        self.prop['deb_wal_fac'] = 0*u.m**2
        self.prop['elastic_c11'] = 327e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c12'] = 128e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c13'] = 103e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c22'] = 327e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c23'] = 103e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c33'] = 351e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['sound_vel'] = 6.3*u.nm/u.ps  # calculated -- np.sqrt(c33/density)
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis'] = ['lambda T: 0.0026e-6*T', 12.7e-6]  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        self.prop['exp_a_axis'] = ['lambda T: 0.0026e-6*T', 12.7e-6]  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        self.prop['exp_b_axis'] = ['lambda T: 0.0026e-6*T', 12.7e-6]  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        self.prop['lin_therm_exp'] = ['lambda T: 0.0041e-6*T', 20.2e-6]  # calculated: exp_c_axis*(1+2*c_13/c_33)
        self.prop['Grun_c_axis'] = [1.35, 1.8]  # 10.1063/1.2902170
        self.prop['Grun_a_axis'] = [1.35, 1.8]  # 10.1063/1.2902170
        self.prop['Grun_b_axis'] = [1.35, 1.8]  # 10.1063/1.2902170
        self.prop['heat_capacity'] = ['lambda T: 0.12*T', 442*u.J/(u.kg * u.K)]
        # 0.08*T (Zahn 2021, 10.1103/PhysRevResearch.3.023032)
        # el: 10.1016/S0301-0104(99)00330-4, ph: 10.1016/0022-3697(81)90174-8
        self.prop['sub_system_coupling'] = [
            'lambda T:3.6e17*(T[1]-T[0])', 'lambda T:3.6e17*(T[0]-T[1])']  # 10.1103/PhysRevB.77.075133
        # self.prop['sub_system_coupling'] = ['17e17*(T_1-T_0)', '17e17*(T_0-T_1)']  # 10.1103/PhysRevResearch.3.023032
        self.prop['therm_cond'] = ['lambda T: (81.4)*(T[0]/T[1])', 9.6]  # 10.1016/S0301-0104(99)00330-4
        self.prop['opt_pen_depth'] = 15*u.nm  # (800nm) estimation
        self.prop['opt_ref_index'] = 2.3223 + 8.8820j  # (800nm, thick film) Werner, J. Phys. Chem. Ref. Data 38 (2009)
        self.prop['opt_ref_index_per_strain'] = 0+0j
        
        # LLB parameters
        self.prop['eff_spin'] = 0.5
        self.prop['curie_temp'] = 630*u.K
        self.prop['lamda'] = 0.005
        self.prop['mag_moment'] = 0.393*u.bohr_magneton
        self.prop['aniso_exponent'] = 3
        self.prop['anisotropy'] = [0.45e6, 0.45e6, 0.45e6]*u.J/u.m**3
        self.prop['exch_stiffness'] = [5e-14, 5e-14, 5e-14]*u.J/u.m
        self.prop['mag_saturation'] = 500e3*u.J/u.T/u.m**3
        self.prop['magneto_elastic_coupling'] = - 7.85e6*u.J/u.m**3
        

    def createUnitCell(self, name, caxis, prop):
        Ni = ud.UnitCell(name, 'Ni', caxis, **prop)
        Ni.add_atom(self.Ni, 0)
        return Ni
    
    

class Ni_111_3TM:
    def __init__(self):

        self.Ni = ud.Atom('Ni')

        self.prop = {}
        self.prop['crystal_struc'] = 'fcc'
        self.prop['curie_temp'] = 634
        self.prop['R_parameter'] = 7.4/1e-12  # Koopmans, Nature Physics (2010) used el-pho coupling
        self.prop['c_axis'] = 2.035*u.angstrom  # periodictable.com -- fcc geometry: 3.524*np.sqrt(3)/3
        self.prop['a_axis'] = 2.3185*u.angstrom  # adjust density
        self.prop['b_axis'] = 2.3185*u.angstrom  # adjust density
        self.prop['molar_mass'] = 58.69*u.g
        self.prop['density'] = 8912*u.kg/(u.m**3)
        self.prop['deb_wal_fac'] = 0*u.m**2
        self.prop['elastic_c11'] = 327e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c12'] = 128e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c13'] = 103e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c22'] = 327e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c23'] = 103e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['elastic_c33'] = 351e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['sound_vel'] = 6.3*u.nm/u.ps  # calculated -- np.sqrt(c33/density)
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis'] = ['lambda T: 0.0026e-6*T', 12.7e-6, 0]  # calculated: Grun*heat_cap*dens/(2c_13+c_33)
        self.prop['exp_a_axis'] = ['lambda T: 0.0026e-6*T', 12.7e-6, 0]  # calculated: Grun*heat_cap*dens/(2c_13+c_33)
        self.prop['exp_b_axis'] = ['lambda T: 0.0026e-6*T', 12.7e-6, 0]  # calculated: Grun*heat_cap*dens/(2c_13+c_33)
        self.prop['lin_therm_exp'] = ['lambda T: 0.0041e-6*T', 20.2e-6, 0]  # calculated: exp_c_axis*(1+2*c_13/c_33)
        self.prop['Grun_c_axis'] = [1.35, 1.8, 0]  # 10.1063/1.2902170
        self.prop['Grun_a_axis'] = [1.35, 1.8, 0]  # 10.1063/1.2902170
        self.prop['Grun_b_axis'] = [1.35, 1.8, 0]  # 10.1063/1.2902170
        self.prop['heat_capacity'] = ['lambda T: 0.12*T', 442*u.J/(u.kg * u.K), 1/8912]
        # 0.08*T (Zahn 2021, 10.1103/PhysRevResearch.3.023032)
        # el: 10.1016/S0301-0104(99)00330-4, ph: 10.1016/0022-3697(81)90174-8
        self.prop['sub_system_coupling'] = ['3.6e17*(T_1-T_0)', '3.6e17*(T_0-T_1)', '{0:f}*T_2*T_1/{1:f}*(1-T_2* (1 + 2/(exp(2*T_2*{1:f}/T_0) - 1) ))'.format(
            self.prop['R_parameter'], self.prop['curie_temp'])]  # 10.1103/PhysRevB.77.075133
        self.prop['therm_cond'] = ['lambda T: (81.4)*(T_0/T_1)', 9.6, 0]  # 10.1016/S0301-0104(99)00330-4
        self.prop['opt_pen_depth'] = 15*u.nm  # (800nm) estimation
        self.prop['opt_ref_index'] = 2.3223 + 8.8820j  # (800nm, thick film) Werner, J. Phys. Chem. Ref. Data 38 (2009)
        self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        Ni = ud.UnitCell(name, 'Ni', caxis, **prop)
        Ni.add_atom(self.Ni, 0)
        return Ni
