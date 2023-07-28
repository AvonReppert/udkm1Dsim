# -*- coding: utf-8 -*-
"""
Created on Fryday Jul 14 22:05:40 2023

@author: Max
"""


import udkm1Dsim as ud
u = ud.u  # import the pint unit registry from udkm1Dsim

import corning_glass as corning_glass
import Pt_111 as Pt_111
import Ni_111 as Ni_111
import Ta_111 as Ta_111
import udkm1Dsim as ud
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import helpers
hel = helpers.helpers()

u.setup_matplotlib()

Pt = Pt_111.Pt_111_2TM()

Ta = Ta_111.Ta_111_2TM()

Ni = Ni_111.Ni_111_LLB()

glass = corning_glass.corning_glass_2TM()




# %%

''' Put In the Simulation Parameter '''
# %% Define Nickel sample


# %%
''' Simulation Parameter '''
# Initialization of the sample
sample_name = 'Ni20'
layers = ['Pt', 'Ni', 'Pt_2', 'Ta', 'glass']
layer_mag = [False, True, False, False, False]
sample_dic = {'Pt': Pt, 'Ni': Ni, 'Pt_2': Pt, 'Ta': Ta, 'glass': glass}
properties = {'Pt': {'C': Pt.prop['heat_capacity']},
              'Ni': {'C': Ni.prop['heat_capacity']},
              'Pt_2': {'C': Pt.prop['heat_capacity']},
              'Ta': {'C': Ta.prop['heat_capacity']},
              'glass': {'C': glass.prop['heat_capacity']}}

# Possible Excitation Conditions of Simulation
angle_list = [90*u.deg]
peak_list = ['Ni']

# Simulated Excitation Conditions
peak_meas = 'Ni'
fluenz_sim = [4/0.682]*u.mJ/u.cm**2
puls_width = [0.05]*u.ps
pump_delay = [0]*u.ps
multi_abs = True
init_temp = 300
heat_diff = True
delays = np.r_[-100:2000:0.05]*u.ps

# Simulation Model
static_exp = True

# Simulation Parameters
num_unit_cell = [9, 98, 13, 10, 100]

# %%
plotting_sim = [True, True, False, False, False]
color_list = ['orange', 'gray', 'black', 'gray', 'red']
data_list = [0, 0, 0, 0, 0]

# %%

''' Build the sample structure from the initialized unit cells '''
# %%

for l in range(len(layers)):
    prop_uni_cell = {}
    
    if layer_mag[l] == True:
        prop_uni_cell['eff_spin'] = sample_dic[layers[l]].prop['eff_spin'] 
        prop_uni_cell['curie_temp'] = sample_dic[layers[l]].prop['curie_temp']
        prop_uni_cell['lamda'] = sample_dic[layers[l]].prop['lamda']
        prop_uni_cell['mag_moment'] = sample_dic[layers[l]].prop['mag_moment'] 
        prop_uni_cell['aniso_exponent'] = sample_dic[layers[l]].prop['aniso_exponent']
        prop_uni_cell['exch_stiffness'] = sample_dic[layers[l]].prop['exch_stiffness']
        prop_uni_cell['mag_saturation'] = sample_dic[layers[l]].prop['mag_saturation']
        
        
    
    prop_uni_cell['a_axis'] = sample_dic[layers[l]].prop['a_axis']
    prop_uni_cell['b_axis'] = sample_dic[layers[l]].prop['b_axis']
    prop_uni_cell['sound_vel'] = sample_dic[layers[l]].prop['sound_vel']
    prop_uni_cell['lin_therm_exp'] = sample_dic[layers[l]].prop['lin_therm_exp']
    prop_uni_cell['heat_capacity'] = sample_dic[layers[l]].prop['heat_capacity']
    prop_uni_cell['therm_cond'] = sample_dic[layers[l]].prop['therm_cond']
    prop_uni_cell['sub_system_coupling'] = sample_dic[layers[l]].prop['sub_system_coupling']
    prop_uni_cell['opt_pen_depth'] = sample_dic[layers[l]].prop['opt_pen_depth']
    prop_uni_cell['opt_ref_index'] = sample_dic[layers[l]].prop['opt_ref_index']
    prop_uni_cell['phonon_damping'] = sample_dic[layers[l]].prop['phonon_damping']
    properties[layers[l]]['unit_cell'] = sample_dic[layers[l]].createUnitCell(
        layers[l], sample_dic[layers[l]].prop['c_axis'], prop_uni_cell)

# LLB parameters
ud.Atom('Ni', mag_amplitude=1, mag_gamma=90*u.deg, mag_phi=0*u.deg)



S = ud.Structure(sample_name)
for l in range(len(layers)):
    S.add_sub_structure(properties[layers[l]]['unit_cell'], num_unit_cell[l])
S.visualize()
print(S)

_, _, distances = S.get_distances_of_layers()


for l in range(len(layers)):
    properties[layers[l]]['num_unit_cell'] = num_unit_cell[l]
    properties[layers[l]]['density'] = properties[layers[l]]['unit_cell'].density
    properties[layers[l]]['select_layer'] = S.get_all_positions_per_unique_layer()[layers[l]]
    properties[layers[l]]['thick_layer'] = properties[layers[l]
                                                      ]['num_unit_cell']*properties[layers[l]]['unit_cell'].c_axis




# %% Get the absorption

h = ud.Heat(S, True)
h.excitation = {'fluence': fluenz_sim, 'delay_pump': pump_delay, 'pulse_width': puls_width,
                'multilayer_absorption': multi_abs, 'wavelength': 800*u.nm, 'theta': angle_list[peak_list.index(peak_meas)]}

dAdzLB = h.get_Lambert_Beer_absorption_profile()
dAdz, _, _, _ = h.get_multilayers_absorption_profile()

'''Plot the absorption profile'''

plt.figure()
plt.plot(distances.to('nm'), 4*dAdz*1e-9*1e2, label=r'4$\cdot$Multilayer')
plt.plot(distances.to('nm'), dAdzLB*1e-9*1e2, label=r'Lambert-Beer')
plt.xlim(0, 100)
plt.legend()
plt.xlabel('Distance (nm)')
plt.ylabel('Differnetial Absorption (%)')
plt.title('Laser Absorption Profile')
plt.show()

# %%

''' Get Temperature Map from the absorption profile including heat diffusion '''

# %%

h.save_data = True
h.disp_messages = True
h.heat_diffusion = heat_diff
h.boundary_conditions = {'top_type': 'isolator', 'bottom_type': 'isolator'}

Init_temp = np.ones([S.get_number_of_layers(), 2])
Init_temp[:, 0] = init_temp
Init_temp[:, 1] = init_temp

temp_map, delta_temp_map = h.get_temp_map(delays, Init_temp)

# %%

Ni_ph_temp = np.mean(temp_map[:, properties['Ni']['select_layer'], 1], axis=1)

plt.figure(figsize=[6, 0.68*6])
plt.plot(delays.to('ps'), Ni_ph_temp)
plt.show()



plt.figure(figsize=[6, 5])
plt.subplot(1, 1, 1)
plt.pcolormesh(distances.to('nm').magnitude, delays.to('ps').magnitude, temp_map[:, :, 1],
               shading='auto', cmap='RdBu_r', vmin=np.min(temp_map[:, :, 1]), vmax=np.max(temp_map[:, :, 1]))
plt.colorbar()
plt.xlim(0, 48)
plt.xlabel('Distance (nm)')
plt.ylabel('Delay (ps)')
plt.title('Phonon')
plt.tight_layout()
plt.show()

plt.figure(figsize=[6, 5])
plt.subplot(1, 1, 1)
plt.pcolormesh(distances.to('nm').magnitude, delays.to('ps').magnitude, temp_map[:, :, 0],
               shading='auto', cmap='RdBu_r', vmin=np.min(temp_map[:, :, 0]), vmax=np.max(temp_map[:, :, 0]))
plt.colorbar()
plt.xlim(0, 48)
plt.ylim(-0.2, 2)
plt.xlabel('Distance (nm)')
plt.ylabel('Delay (ps)')
plt.title('Electron')
plt.tight_layout()
plt.show()


# %%

# %%

pnum = ud.PhononNum(S, True)
hnum = ud.PhononNum(S, True, only_heat  = True )
pnum.save_data = True
pnum.disp_messages = True

strain_map = pnum.get_strain_map(delays, temp_map, delta_temp_map)
#%%
strain_map_h = hnum.get_strain_map(delays, temp_map, delta_temp_map)

#strain_map_h = strain_map

#%%Weigh the strain with the absorption proile

# def weigh_strain_with_normalized_profile(strain_map, profile):
#     normalized_profile = profile / np.sum(profile)
#     weighted_strain = np.dot(strain_map, normalized_profile)
#     return weighted_strain


# dAdzNi = dAdz[S.get_all_positions_per_unique_layer()['Ni']]

# strain_map_Ni = strain_map[:, properties['Ni']['select_layer']]


# weighed_strain = weigh_strain_with_normalized_profile(strain_map_Ni, dAdzNi)

# plt.figure(figsize=[6, 0.68*6])
# #plt.pcolormesh(distances.to('nm').magnitude, delays.to('ps').magnitude, 1e3*strain_map, shading='auto',
#                 cmap='RdBu_r', vmin=-0.7*np.max(1e3*strain_map), vmax=0.7*np.max(1e3*strain_map))
# plt.colorbar()
# plt.xlim(0, 140)
# plt.ylim(-2, 400)
# plt.xlabel('Distance (nm)')
# plt.ylabel('Delay (ps)')
# plt.title(r'Strain Map ($10^{-3}$)')
# plt.tight_layout()
# plt.show()


# Ni_strain = np.mean(strain_map[:, properties['Ni']['select_layer']], axis=1)

# Ni_heat_strain = np.mean(strain_map_h[:, properties['Ni']['select_layer']], axis=1)

# plt.figure(figsize=[6, 0.68*6])
# plt.plot(delays.to('ps'), Ni_strain)
# plt.plot(delays.to('ps'), Ni_heat_strain)
# plt.plot(delays.to('ps'), weighed_strain, color = 'k')
# plt.show()



temp_map_hom = np.zeros_like(temp_map)

temp_map_hom += 300

idt = np.where(delays > 0)[0][0]



plt.pcolormesh(distances.to('nm').magnitude, delays.to('ps').magnitude, 1e3*strain_map_h, shading='auto',
                cmap='RdBu_r', vmin=-0.7*np.max(1e3*strain_map_h), vmax=0.7*np.max(1e3*strain_map))
plt.colorbar()
plt.xlim(0, 40)
plt.ylim(-50, 400)
plt.xlabel('Distance (nm)')
plt.ylabel('Delay (ps)')
plt.title(r'Strain Map ($10^{-3}$)')
plt.tight_layout()
plt.show()


#%% LLB Calculations

llg = ud.LLG(S, True)

llg.save_data = False
llg.disp_messages = True

print(llg)



angles = np.linspace(np.pi/16, np.pi/2, 1)

m_0_l = 0.979255661045889
m_0_0 = 0.846729079775114
m_0_1 = 0.0017708926455357985


for angle in angles:


    init_mag = np.array([m_0_l , m_0_0, m_0_1])
    
    magnetization_map = llg.get_magnetization_map(delays, temp_map=temp_map, strain_map=strain_map_h, H_ext=np.array([0.4*np.sin(angle), 0., 0.4*np.cos(angle)]), init_mag=init_mag)
    
    plt.figure(figsize=[6, 12])
    plt.subplot(3, 1, 1)
    
    plt.xlim(0,22)
    plt.ylim(-5,250)
    
    
    plt.pcolormesh(distances.to('nm').magnitude, delays.to('ps').magnitude, magnetization_map[:, :, 0],
                    shading='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel('Distance [nm]')
    plt.ylabel('Delay [ps]')
    plt.title(f'Amplitude at {np.round(angle*180/np.pi, 0)} °')
    
    plt.subplot(3, 1, 2)
    
    plt.xlim(0,22)
    plt.ylim(-5,250)
    
    plt.pcolormesh(distances.to('nm').magnitude, delays.to('ps').magnitude, magnetization_map[:, :, 1],
                    shading='auto', cmap='RdBu_r', vmin=-3.14, vmax=3.14)
    plt.colorbar()
    plt.xlabel('Distance [nm]')
    plt.ylabel('Delay [ps]')
    plt.title('$\phi$')
    
    plt.subplot(3, 1, 3)
    
    plt.xlim(0,22)
    plt.ylim(-5,250)
    
    plt.pcolormesh(distances.to('nm').magnitude, delays.to('ps').magnitude, magnetization_map[:, :, 2],
                    shading='auto', cmap='RdBu_r', vmin=-3.14, vmax=3.14)
    plt.colorbar()
    plt.xlabel('Distance [nm]')
    plt.ylabel('Delay [ps]')
    plt.title('$\gamma$')
    
    
    
    plt.tight_layout()
    plt.show()
    
    
    plt.figure(figsize=[6,8])
    plt.subplot(2,1,1)
    plt.plot(delays, np.mean(magnetization_map[:, 9:108, 0], axis=1), label=r'$A$')
    plt.legend()
    plt.xlabel('Delay (ps)')
    plt.ylabel('Magnetization')
    plt.subplot(2,1,2)
    plt.plot(delays, (np.mean(magnetization_map[:, 9:108, 1], axis=1)*u.rad).to('deg'), label=r'$\phi$')
    plt.plot(delays, (np.mean(magnetization_map[:, 9:108, 2], axis=1)*u.rad).to('deg'), label=r'$\gamma$')
    plt.legend()
    plt.xlabel('Delay (ps)')
    plt.ylabel('Magnetization')
    plt.show()
    
    #%%
    magnetization_map_xyz = ud.helpers.convert_polar_to_cartesian(magnetization_map)

    m_x_max = 1.2 * np.max(magnetization_map_xyz[idt:, 9:108, 0])
    m_y_max = 1.2 * np.max(magnetization_map_xyz[idt:, 9:108, 1])
    m_z_max = 1.2 * np.max(magnetization_map_xyz[idt:, 9:108, 2])
    
    m_x_min = 1.2 * np.min(magnetization_map_xyz[idt:, 9:108, 0])
    m_y_min = 1.2 * np.min(magnetization_map_xyz[idt:, 9:108, 1])
    m_z_min = 1.2 * np.min(magnetization_map_xyz[idt:, 9:108, 2])
    
    plt.figure(figsize=[6, 12])
    plt.subplot(3, 1, 1)
    plt.pcolormesh(distances.to('nm').magnitude, delays.to('ps').magnitude, magnetization_map_xyz[:, :, 0],
                   shading='auto', cmap='RdBu', vmin = m_x_min, vmax = m_x_max )
    plt.colorbar()
    plt.xlabel('Distance [nm]')
    plt.ylabel('Delay [ps]')
    plt.title('$M_x$')
    plt.xlim(0,22)
    plt.ylim(-5,50)
    plt.subplot(3, 1, 2)
    plt.pcolormesh(distances.to('nm').magnitude, delays.to('ps').magnitude, magnetization_map_xyz[:, :, 1],
                   shading='auto', cmap='RdBu', vmin = m_y_min, vmax = m_y_max)
    plt.colorbar()
    plt.xlabel('Distance [nm]')
    plt.ylabel('Delay [ps]')
    plt.title('$M_y$')
    plt.xlim(0,22)
    plt.ylim(-5,50)
    plt.subplot(3, 1, 3)
    plt.pcolormesh(distances.to('nm').magnitude, delays.to('ps').magnitude, magnetization_map_xyz[:, :, 2],
                   shading='auto', cmap='RdBu',  vmin = m_z_min, vmax = m_z_max )
    plt.colorbar()
    plt.xlabel('Distance [nm]')
    plt.ylabel('Delay [ps]')
    plt.title('$M_z$')
    plt.xlim(0,22)
    plt.ylim(-5,500)
    plt.tight_layout()
    plt.show()
    
    
    plt.figure(figsize=[6,8])
    plt.title(f'Magnetization dynamics at {np.round(angle*180/np.pi, 0)} °')
    plt.subplot(3,1,1)
    plt.xlim(-50,800)
    plt.ylim(m_x_min,m_x_max)
    plt.plot(delays, np.mean(magnetization_map_xyz[:, 9:108, 0], axis=1), label=r'$M_x$ Ni')
    plt.legend()
    plt.xlabel('Delay (ps)')
    plt.ylabel('Magnetization')
    plt.subplot(3,1,2)
    plt.xlim(-50,800)
    plt.ylim(m_y_min,m_y_max)
    plt.plot(delays, np.mean(magnetization_map_xyz[:, 9:108, 1], axis=1), label=r'$M_y$ Ni')
    plt.legend()
    plt.subplot(3,1,3)
    plt.xlim(-50,800)
    plt.ylim(m_z_min,m_z_max)
    plt.plot(delays, np.mean(magnetization_map_xyz[:, 9:108, 2], axis=1), label=r'$M_z$ Ni')
    plt.legend()
    plt.xlabel('Delay (ps)')
    plt.ylabel('Magnetization')
    plt.show()

    plt.figure(figsize=[6,8])
    plt.title(f'Magnetization dynamics at {np.round(angle*180/np.pi, 0)} °')
    plt.subplot(3,1,1)
    plt.plot(delays, np.mean(magnetization_map_xyz[:, 9:108, 0], axis=1), label=r'$M_x$ Ni')
    plt.legend()
    plt.xlabel('Delay (ps)')
    plt.ylabel('Magnetization')
    plt.subplot(3,1,2)
    plt.plot(delays, np.mean(magnetization_map_xyz[:, 9:108, 1], axis=1), label=r'$M_y$ Ni')
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(delays, np.mean(magnetization_map_xyz[:, 9:108, 2], axis=1), label=r'$M_z$ Ni')
    plt.legend()
    plt.xlabel('Delay (ps)')
    plt.ylabel('Magnetization')
    plt.show()