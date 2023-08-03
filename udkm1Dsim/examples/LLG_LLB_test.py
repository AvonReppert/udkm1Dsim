import matplotlib.pyplot as plt
import numpy as np
import udkm1Dsim as ud
u = ud.u  # import the pint unit registry from udkm1Dsim
u.setup_matplotlib()  # use matplotlib with pint units


# %% Layer properties

# Loading atoms
Ni = ud.Atom('Ni', mag_amplitude=1, mag_gamma=90*u.deg, mag_phi=0*u.deg)
Si = ud.Atom('Si')

prop_Ni = {}
# parameters for a two-temperture model
prop_Ni['heat_capacity'] = ['0.1*T',         532*u.J/u.kg/u.K, ]
prop_Ni['therm_cond'] = [20*u.W/(u.m*u.K),  80*u.W/(u.m*u.K), ]
g = 4.0e18  # electron-phonon coupling
prop_Ni['sub_system_coupling'] = \
  ['-{:f}*(T_0-T_1)'.format(g),
   '{:f}*(T_0-T_1)'.format(g)
   ]
prop_Ni['lin_therm_exp'] = [0, 11.8e-6]
prop_Ni['opt_ref_index'] = 2.9174+3.3545j

# LLB parameters
prop_Ni['eff_spin'] = 0.5
prop_Ni['curie_temp'] = 630*u.K
prop_Ni['lamda'] = 0.0005
prop_Ni['mag_moment'] = 0.393*u.bohr_magneton
prop_Ni['aniso_exponent'] = 3
prop_Ni['anisotropy'] = [0.45e6, 0.45e6, 0.45e6]*u.J/u.m**3
prop_Ni['exch_stiffness'] = [0.1e-15, 1e-15, 0.01e-15]*u.J/u.m
prop_Ni['mag_saturation'] = 500e3*u.J/u.T/u.m**3

# build the layer
layer_Ni = ud.AmorphousLayer('Ni', 'Ni amorphous', thickness=1*u.nm,  density=7000*u.kg/u.m**3, atom=Ni, **prop_Ni)


prop_Si = {}
prop_Si['heat_capacity'] = [100*u.J/u.kg/u.K, 603*u.J/u.kg/u.K]
prop_Si['therm_cond'] = [0, 100*u.W/(u.m*u.K)]
prop_Si['sub_system_coupling'] = [0, 0]
prop_Si['lin_therm_exp'] = [0, 2.6e-6]
prop_Si['sound_vel'] = 8.433*u.nm/u.ps
prop_Si['opt_ref_index'] = 3.6941+0.0065435j

layer_Si = ud.AmorphousLayer('Si', "Si amorphous", thickness=1*u.nm, density=2336*u.kg/u.m**3,
                           atom=Si, **prop_Si)


# %% Building the structure

S = ud.Structure('Ni')
S.add_sub_structure(layer_Ni, 20)
S.add_sub_structure(layer_Si, 200)
S.visualize()

# %% Heat simulation with excitation
init_temp = 300  # K
fluence = 15     # mJ/cm^2
_, _, distances = S.get_distances_of_layers()


h = ud.Heat(S, force_recalc=False)
h.save_data = True
h.disp_messages = True
delays = np.r_[-100:200:1]*u.ps
h.excitation = {'fluence': [fluence]*u.mJ/u.cm**2,
              'delay_pump':  [0]*u.ps,
              'pulse_width':  [0.15]*u.ps,
              'multilayer_absorption': True,
              'wavelength': 800*u.nm,
              'theta': 45*u.deg}

# enable heat diffusion
h.heat_diffusion = True
# set the boundary conditions
h.boundary_conditions = {'top_type': 'isolator', 'bottom_type': 'isolator'}
# The resulting temperature profile is calculated in one line:

temp_map, delta_temp = h.get_temp_map(delays, init_temp)

pnum = ud.PhononNum(S, True)
strain_map = pnum.get_strain_map(delays, temp_map, delta_temp)

is_magnetic = S.get_layer_property_vector('_curie_temp')>0
# %%
llb = ud.LLB(S, True)

llb.save_data = True
llb.disp_messages = True

H_ext = np.array([1,0,1])


init_mag_map = llb.calc_equilibrium_magnetization_map(H_ext=H_ext, threshold = 1e-3)


magnetization_map = llb.get_magnetization_map(delays, temp_map=temp_map, strain_map = strain_map, init_mag=init_mag_map, H_ext=H_ext)
magnetization_map_xyz = ud.LLB.convert_polar_to_cartesian(magnetization_map)


plt.figure(figsize=[6, 8])
plt.subplot(2, 1, 1)
plt.plot(delays, np.mean(magnetization_map_xyz[:, is_magnetic, 2], axis=1), label=r'$M_z$')
plt.legend()
plt.xlabel('Delay (ps)')
plt.ylabel('Magnetization')
plt.subplot(2, 1, 2)
plt.plot(delays, np.mean(magnetization_map_xyz[:, is_magnetic, 0], axis=1), label=r'$M_x$')
plt.plot(delays, np.mean(magnetization_map_xyz[:, is_magnetic, 1], axis=1), label=r'$M_y$')
plt.legend()
plt.xlabel('Delay (ps)')
plt.ylabel('Magnetization')
plt.show()

# %%
llg = ud.LLG(S, True)

llg.save_data = True
llg.disp_messages = True


init_mag_map = llg.calc_equilibrium_magnetization_map(H_ext=H_ext, threshold = 1e-5)


magnetization_map = llg.get_magnetization_map(delays, temp_map=temp_map, strain_map = strain_map, init_mag=init_mag_map, H_ext=H_ext)
magnetization_map_xyz = ud.LLB.convert_polar_to_cartesian(magnetization_map)


plt.figure(figsize=[6, 8])
plt.subplot(2, 1, 1)
plt.plot(delays, np.mean(magnetization_map_xyz[:, is_magnetic, 2], axis=1), label=r'$M_z$')
plt.legend()
plt.xlabel('Delay (ps)')
plt.ylabel('Magnetization')
plt.subplot(2, 1, 2)
plt.plot(delays, np.mean(magnetization_map_xyz[:, is_magnetic, 0], axis=1), label=r'$M_x$')
plt.plot(delays, np.mean(magnetization_map_xyz[:, is_magnetic, 1], axis=1), label=r'$M_y$')
plt.legend()
plt.xlabel('Delay (ps)')
plt.ylabel('Magnetization')
plt.show()