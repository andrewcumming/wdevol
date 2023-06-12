import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy import interpolate
from eos import *
  
# ----------------------------------------------------------------------------

def integrate(rho_c):
    # integrate the white dwarf structure for central density rho_c
    # We use this to get an initial guess for the hydrostatic structure
    # and to check the hydrostatic structure found by the Henyey code

    # to store the profiles
    rho_vec = np.array([])
    P_vec = np.array([])
    m_vec = np.array([])
    r_vec = np.array([])
    gam_vec = np.array([])

    # step in r
    dr = 1e6 # cm

    # initial conditions
    m = 2e28  # 1e-5 of a solar mass
    r = (3 * m / (4 * np.pi * rho_c)) ** (1/3)
    rho = rho_c
    dlnPdlnrho = gamma(rho_c, 0.0, X_interp(0.0))
    P = pressure(rho, 0.0, X_interp(0.0))

    # take steps until we reach the surface
    while P > P_0:
    #while rho > rho_0:
    
        # store the latest step
        rho_vec = np.append(rho_vec,rho)    
        P_vec = np.append(P_vec,P)    
        r_vec = np.append(r_vec,r)    
        m_vec = np.append(m_vec,m)    
        gam_vec = np.append(gam_vec,dlnPdlnrho)    
      
        # calculate dP/drho
        P = pressure(rho, 0.0, X_interp(m/mass))
        dlnPdlnrho = gamma(rho, 0.0, X_interp(m/mass))

        # derivatives    
        dmdr = 4.0 * np.pi * r**2 * rho
        dPdr = - Ggrav * m * rho / r**2
        drhodr = (rho/P) * dPdr / dlnPdlnrho 
   
        # take a step
        r = r + dr
        m = m + dmdr * dr
        rho = rho + drhodr * dr
    
    return rho_vec, m_vec, r_vec, P_vec

# ----------------------------------------------------------------------------

def residual(y):
    # calculate the finite-difference residuals for the current model

    # extract the new values
    if delta_t == 0.0:
        Pn, rn, Ln, Tn, LXn, Xn, mn = np.split(y, 7)
    else:
        Pn, rn, Ln, Tn, LXn, Xn = np.split(y, 6)
        mn = m
            
    rhon = 10.0**rho_interp((np.log10(Pn),Xn))
    K = 10.0**K_interp((np.log10(rhon),Xn))
    CP = 10.0**CP_interp((np.log10(rhon),Xn))
    
    KX = KX_scale * np.ones_like(K)

    # quantities at the 1/2 grid points
    dm = mn[1:]-mn[:-1]
    m12 = 0.5 * (mn[1:] + mn[:-1])
    dm12 = m12[1:]-m12[:-1]
    dm12 = np.append(dm12,dm[-1])
    r12 = 0.5 * (rn[1:] + rn[:-1])
    rho12 = 0.5 * (rhon[1:] + rhon[:-1])
    P12 = 0.5 * (Pn[1:] + Pn[:-1])
    X12 = 0.5 * (Xn[1:] + Xn[:-1])
    K12 = 0.5 * (K[1:] + K[:-1])

    mn_solid=0.0

    if delta_t == 0.0:   # we're trying to find the initial model, so assume uniform eps_grav
        eps_grav = np.ones_like(Pn) * (Ltop0/mass)
        L_surf = Ltop0
        T_surf = 10**Ttop_interp(np.log10(Ltop0/Lsun))
        eps_X = np.ones_like(Xn) * (LXn[-1]/mass)
    else:  # compute eps_grav based on the timestep
        dT = Tn-T
        drho = rhon-rho
        dX = Xn-X
        
        T_surf = 10**Ttop_interp(np.log10(Ln[-1]/Lsun))
        
        # In the next line, we set (1-chi_T del_ad)\approx 1 in the dT/dt term
        # and approximate chi_rho=4/3 and del_ad=1/3 in the drho/dt term
        eps_grav = -CP * dT / delta_t + (1/3)*(4/3)*CP*(Tn/rhon)*drho/delta_t
        eps_X = - dX/delta_t
    
        # solid core mass  
        if gamma_av(rhon[0],Tn[0],Xn[0]) < gamma_melt:
            mn_solid = 0.0
        elif gamma_av(rhon[-1],Tn[-1],Xn[-1]) > gamma_melt:
            mn_solid = mn[-1]
        else:
            gamma = np.zeros_like(rhon)
            for i in range(len(gamma)):
                gamma[i] = gamma_av(rhon[i], Tn[i], Xn[i])
            gamma_interp = interpolate.interp1d(gamma, mn/mass)
            mn_solid = mass * gamma_interp(gamma_melt)
            QL = 0.77 * (Xn/12 + (1-Xn)/16) * Tn * (1.38e-16/1.67e-24)
            QX = -np.ones_like(QL) * DXc * np.minimum(20*mn_solid/mass, 1.0)
            #QX = np.zeros_like(QL)  # turn off chemical separation
            
            # identify the cells that contain mass that has solidified in this timestep
            ind = np.where(np.logical_and(mn<mn_solid, mn>=m_solid))
            ind = ind[0]
            
            nliq = 4 # number of cells in the liquid that receive the light elements
            #print("Solid formed in cells: ", ind)
            if len(ind)==0:   # no cells match, so the solid was formed inside one cell
                #print('no cell')
                ind = np.where(m<mn_solid)
                ind = ind[0]
                ii = ind[-1]
                eps_grav[ii] += QL[ii]/delta_t * (mn_solid-m_solid)/dm[ii]
                eps_X[ii] += QX[ii]/delta_t * (mn_solid-m_solid)/dm[ii]
                # liquid region                
                eps_X[ii+1:ii+1+nliq] += -QX[ii+1:ii+1+nliq]/delta_t * (mn_solid-m_solid)/(np.sum(dm[ii+1:ii+1+nliq]))
            else:
                #first cell
                ii = ind[0]
                if ii>0:
                    eps_grav[ii-1] += QL[ii-1]/delta_t * (m[ii]-m_solid)/dm[ii-1]
                    eps_X[ii-1] += QX[ii-1]/delta_t * (m[ii]-m_solid)/dm[ii-1]
                # middle cells if needed
                if len(ind)>1:
                    eps_grav[ind[:-1]] += QL[ind[:-1]]/delta_t
                    eps_X[ind[:-1]] += QX[ind[:-1]]/delta_t
                # last cell
                ii = ind[-1]
                eps_grav[ii] += QL[ii]/delta_t * (mn_solid-m[ii])/dm[ii]
                eps_X[ii] += QX[ii]/delta_t * (mn_solid-m[ii])/dm[ii]
                # and put the light elements back in the liquid
                eps_X[ii+1:ii+1+nliq] += -QX[ii+1:ii+1+nliq]/delta_t * (mn_solid-m_solid)/(np.sum(dm[ii+1:ii+1+nliq]))
                  
    KX12 = 0.5 * (KX[1:] + KX[:-1])
    
    # enhanced diffusion in the liquid phase
    ind = m12 >= mn_solid
    KX12[ind] = KX_factor * KX_scale * (1.0 - ((m12[ind]-mn_solid)/(mass-mn_solid))**2)
          
    # hydrostatic balance
    f1 = np.zeros_like(Pn)        
    f1[:-1] = Pn[1:]-Pn[:-1] + dm12 * Ggrav * mn[1:] / (4*np.pi * rn[1:]**4)
    f1[-1] = P_0 - Pn[-1]   # fixed pressure outer boundary
    f1 = f1/Pn  # rescale to order unity

    # mass conservation
    f2 = np.zeros_like(rn)
    #f2[1:] = r[1:]-r[:-1] - dm / (4*np.pi*r12**2*rho12)
    # write this in the form of eq. (5) in MESA I
    f2[1:] = np.log(rn[1:])-np.log(rn[:-1]**3 + 3 * dm / (4*np.pi*rhon[:-1]))/3
    f2[0] = rn[0]/rn[-1]  # r=0 at the centre

    # luminosity
    f4 = np.zeros_like(Ln)
    f4[1:] = Ln[:-1]-Ln[1:] + dm * eps_grav[:-1]
    f4[0] = Ln[0]      # L=0 at the center
    f4 = f4/Ln[-1]

    # temperature
    f5 = np.zeros_like(Tn)
    f5[1:] = Tn[1:]-Tn[:-1] + dm12 * Ln[1:]/((4*np.pi*rn[1:]**2)**2 * K12 * rho12)
    f5[0] = Tn[-1] - T_surf   # surface temperature (as determined by the luminosity)
    f5 = f5/Tn[0] 
    
    # LX
    f6 = np.zeros_like(LXn)
    f6[1:] = LXn[:-1]-LXn[1:] + dm * eps_X[:-1]
    f6[0] = LXn[0]
    f6 = f6/LXn[-1]
    
    # X
    f7 = np.zeros_like(Xn)
    
    delx = (LXn[1:]/mn[1:]) * P12 / (4.0*np.pi*Ggrav * rho12 * KX12 * X12)
    delx[mn[1:]>=mn_solid] = np.minimum(delx[mn[1:]>=mn_solid], 0.01)
    f7[1:] = Xn[1:]-Xn[:-1] + dm12 * delx * (X12/P12) * Ggrav * mn[1:] / (4*np.pi * rn[1:]**4)

    #f7[1:] = Xn[1:]-Xn[:-1] + dm12 * LXn[1:] / ((4*np.pi*rn[1:]**2)**2 * KX12 * rho12)
    
    f7[0] = Xn[-1] - X_interp(mass)

    if delta_t == 0.0:
        # mesh spacing
        f3 = np.zeros_like(mn)
        #ff = (mn/mass)
        ff = (mn/mass)**(2/3) - np.log(Pn)/ngrid
        f3[1:-1] = ff[2:] + ff[:-2] - 2*ff[1:-1]
        f3[0] = mn[0]    # m=0 at the center
        f3[-1] = 1 - mn[-1]/mass     # m=M at the surface
    
        new_y = np.concatenate((f1, f2, f4, f5, f6, f7, f3), axis=None)

    else:

        new_y = np.concatenate((f1, f2, f4, f5, f6, f7), axis=None)

    return new_y


# ----------------------------------------------------------------------------

# make interpolating functions for rho, CP, K
Xvec = np.linspace(0.0,1.0,100)
rhovec = np.linspace(2.0,10.0,1000)
Karr = np.zeros((1000,100))
CParr = np.zeros((1000,100))
for i in range(len(rhovec)):
    for j in range(len(Xvec)):
        Karr[i,j] = np.log10(thermal_conductivity(10**rhovec[i], 1e6, Xvec[j]))
        CParr[i,j] = np.log10(heat_capacity(10**rhovec[i], 0.0, Xvec[j]))
K_interp = interpolate.RegularGridInterpolator((rhovec,Xvec),Karr,bounds_error=False, fill_value = None)
CP_interp = interpolate.RegularGridInterpolator((rhovec,Xvec),CParr,bounds_error=False, fill_value = None)

Pvec = np.linspace(18.0,25.0,1000)
rhoarr = np.zeros((1000,100))
for i in range(len(Pvec)):
    for j in range(len(Xvec)):
        rhoarr[i,j] = np.log10(find_rho(10**Pvec[i],0.0,Xvec[j]))
rho_interp = interpolate.RegularGridInterpolator((Pvec,Xvec),rhoarr,bounds_error=False, fill_value = None)

#rho_interp = interpolate.interp1d(Pvec,rhovec, bounds_error=False, fill_value = 'extrapolate')
#K_interp = interpolate.interp1d(rhovec,Kvec, bounds_error=False, fill_value = 'extrapolate')
#CP_interp = interpolate.interp1d(rhovec,CPvec, bounds_error=False, fill_value = 'extrapolate')

# ----------------------------------------------------------------------------

# constants
Ggrav = 6.67e-8
Lsun = 3.828e33
Msun = 1.989e33
secperyear = 3.15e7

# melting criterion
gamma_melt = 175.0
DXc = 0.1

# white dwarf mass
mass = 0.8 * Msun

# grid size
ngrid = 30

# composition
mvec = np.linspace(0.0,1.0,100)
Xvec = 0.5 + 1e-3 * mvec
#Xvec = 0.5 + 0.0001 * (1.0 - mvec**2)
X_interp = interpolate.interp1d(mvec,Xvec,bounds_error=False, fill_value = (Xvec[0],Xvec[-1]))

KX_scale = 1e3 # 1e6 gives a diffusion time ~ Hubble time
KX_factor = 1e3  # factor by which the diffusion is larger in the liquid phase

# pressure at outer boundary
P_0 = 1e21

# Read in the Ltop-Ttop relation, L(T) at P=1e21 cgs
# These are tabulated from MESA for masses 0.3, 0.4... 0.9, 1.0 solar masses
line_to_read = int((mass-0.3)/0.1)
Ttop_vals = np.loadtxt('data/Tb21.txt', skiprows=2, max_rows=1, delimiter=',')
Ltop_vals = np.loadtxt('data/log_L21.txt', skiprows=2, max_rows=1, delimiter=',')
Ttop_interp = interpolate.interp1d(Ltop_vals,Ttop_vals)
print(Ltop_vals, Ttop_vals)

# initial luminosity and corresponding temperature at the outer boundary
Ltop0 = 1e-3 * Lsun
Ttop0 = 10**Ttop_interp(np.log10(Ltop0/Lsun))

# store the time history
time = 0.0
L_hist = np.array([Ltop0/Lsun])
t_hist = np.array([time])
res_hist = np.array([])
deltat_hist = np.array([])
model_hist = np.array([1])

# ----------------------------------------------------------------------------

print('----------- Looking for initial model -------------')

# initial mass values
m_guess = np.linspace(0.0, mass, num=ngrid)

# trial solution: guess central density
rho_c = 3e7

# Constant density star:
# (this works, but gives a much slower initial step because 
# it's further away from the correct solution)
#P_c = (2*np.pi/3)*Ggrav*rho_c**2 * (3*mass/(4*np.pi*rho_c))**(2/3)
#r_guess = (3 * m_guess / (4*np.pi*rho_c))**(1/3)
#P_guess = P_c - Ggrav * (2*np.pi/3)*(3*m_guess*rho_c**2/(4.0*np.pi))**(2/3)

# Do an integration with this central density
rhov, mv, rv, Pv = integrate(rho_c)
rho_guess = np.interp(m_guess, mv, rhov)
r_guess = np.interp(m_guess, mv, rv)
#print(m_guess, r_guess, mv, rv)
print(max(m_guess), max(mv))
P_guess = np.interp(m_guess, mv, Pv)
T_guess = Ttop0 * np.ones_like(P_guess)
L_guess = Ltop0 * m_guess/mass
LX_guess = np.zeros_like(P_guess)
X_guess = X_interp(m_guess/mass)
r12_guess = 0.5 * (r_guess[1:] + r_guess[:-1])
rho12_guess = 0.5 * (rho_guess[1:] + rho_guess[:-1])
LX_guess[1:] = -(X_guess[1:]-X_guess[:-1])/(m_guess[1:]-m_guess[:-1])*(4*np.pi*r12_guess**2)**2 * KX_scale * rho12_guess
guess = np.concatenate((P_guess, r_guess, L_guess, T_guess, LX_guess, X_guess, m_guess), axis=None)

# solve
delta_t = 0.0
sol = optimize.root(residual, guess, method='lm')
res = residual(sol.x)
print('Residual: %g' % abs(res).max())
res_hist = np.append(res_hist, abs(res).max())

# update the structure
y = sol.x
P, r, L, T, LX, X, m = np.split(y, 7)
rho = 10.0**rho_interp((np.log10(P),X))

print('Initial model:')
print('R_9= ', r[-1]/1e9)
print('rho_c,6 = ', rho[0]/1e6)
print('P_c,23 = ', P[0]/1e23)

print('---------------------------------------------------')

# reset the composition profile
X = X_interp(m/mass)
m_solid = 0.0   # assumes we always start off hot enough to have no solid core

# do an integration with the same central density as a comparison
rhov, mv, rv, Pv = integrate(rho[0])

# Initial time step
delta_t = 1e6*secperyear
deltat_hist = np.append(deltat_hist, delta_t/secperyear)

# initialize the plots
plt.ion()
fig = plt.figure(figsize=(12,9))

fig.suptitle(r'$\mathrm{Model}\ %d\ \ \ t=%g\ \mathrm{Gyr}\ \ \ L=%.2e\ L_\odot$' % (1, 0, Ltop0/Lsun))

ax = plt.subplot(331)
plt.plot(mv/Msun,rv, 'g--')
plt.plot(m/Msun, r, 'r:')
plot_handle_mr, = plt.plot(m/Msun, r, 'o')
plt.xlabel(r'$m\ (M_\odot)$')
plt.ylabel(r'$r\ (\mathrm{cm})$')

ax = plt.subplot(332)
plt.plot(rv, rhov, 'g--')
plt.plot(r, rho, 'r:')
plot_handle_rrho, = plt.plot(r, rho, 'o')
ax.set_yscale('log')
plt.ylabel(r'$\rho\ (\mathrm{g\ cm^{-3}})$')
plt.xlabel(r'$r\ (\mathrm{cm})$')

ax = plt.subplot(333)
t0,L0 = np.loadtxt('history0.dat', unpack=True)
plt.plot(t0,L0, 'r:')
t0,L0 = np.loadtxt('history_melt100.dat', unpack=True)
plt.plot(t0,L0, 'g:')
plot_handle_L, = plt.plot(t_hist/secperyear, L_hist/Lsun)
plt.xlim((1e7, 2e10))
plt.ylim((1e-6, 2e-3))
ax.set_yscale('log')
ax.set_xscale('log')
plt.ylabel(r'$L\ (L_\odot)$')
plt.xlabel(r'$t (\mathrm{yr})$')


ax_res = plt.subplot(336)
plot_handle_res, = plt.plot(model_hist, res_hist)
#plt.xlim((1e7, max(t_hist/secperyear)))
#plt.ylim((min(res_hist),max(res_hist)))
#plt.xlim((1e7, 2e10))
ax_res.set_yscale('log')
#ax_res.set_xscale('log')
plt.ylabel(r'$\mathrm{Residual}$')
#plt.xlabel(r'$t (\mathrm{yr})$')
plt.xlabel(r'$\mathrm{Model\ number}$')


ax_deltat = plt.subplot(339)
plot_handle_deltat, = plt.plot(model_hist, delta_t/secperyear)
#plt.xlim((1e7, max(t_hist/secperyear)))
#plt.ylim((min(res_hist),max(res_hist)))
#plt.xlim((1e7, 2e10))
ax_deltat.set_yscale('log')
#ax_deltat.set_xscale('log')
plt.ylabel(r'$\mathrm{Timestep}\ (\mathrm{yr})$')
plt.xlabel(r'$\mathrm{Model\ number}$')


ax = plt.subplot(334)
Lmax = np.max(L/Lsun)
plot_handle_solid2, = plt.plot([0.0,0.0], [1e-4*Lmax,Lmax], 'g:')
plot_handle_mL, = plt.plot(m/Msun, L/Lsun, 'o')
#plt.plot(mv,rv, '--')
plt.ylim((1e-4*Lmax, Lmax))
ax.set_yscale('log')
plt.xlabel(r'$m\ (M_\odot)$')
plt.ylabel(r'$L\ (L_\odot)$')

ax = plt.subplot(335)
plot_handle_T, = plt.plot(P,T, 'o')
plt.ylim((2e6,3e7))
#plt.plot(rv, rhov, '--')
ax.set_yscale('log')
ax.set_xscale('log')
plt.ylabel(r'$T\ (\mathrm{K})$')
#plt.xlabel(r'$r\ (\mathrm{cm})$')
plt.xlabel(r'$P\ (\mathrm{erg\ cm^{-3}})$')
#plt.xlabel(r'$m\ (\mathrm{g})$')

ax = plt.subplot(338)
plt.plot(m/Msun, X, 'r:')
mm = np.arange(0.0,0.8,0.01)
XX = 0.5 - 0.1*(1.0 + np.log(1.0 - mm))
plt.plot(mm*mass/Msun,XX, 'b:')
plot_handle_X, = plt.plot(m/Msun, X, 'o')
plot_handle_solid, = plt.plot([0.0,0.0], [min(X)-0.15,max(X)+0.15], 'g:')
plt.xlabel(r'$m\ (M_\odot))$')
plt.ylabel(r'$X$')

ax = plt.subplot(337)
Lmax = np.max(abs(LX))
plot_handle_solid3, = plt.plot([0.0,0.0], [1e-4*Lmax,1e5*Lmax], 'g:')
ind = LX<0.0
plot_handle_mLX, = plt.plot(m[ind]/Msun, abs(LX[ind]), 'x')
ind = LX>0.0
plot_handle_mLX2, = plt.plot(m[ind]/Msun, abs(LX[ind]), 'C0o')
#plt.plot(mv,rv, '--')
plt.ylim((1e-1*Lmax, 1e5*Lmax))
ax.set_yscale('log')
plt.xlabel(r'$m\ (M_\odot)$')
plt.ylabel(r'$L_X$')

plt.tight_layout()

fig.canvas.draw()

#print('Press enter to start integration')
#input()

# now take time steps
n_steps = 1000
max_time = 9e9*secperyear

count = 1

while count < n_steps and time < max_time and m_solid/mass < 1.0:

    guess = np.concatenate((P, r, L, T, LX, X), axis=None)
    # if needed can set the tolerance with options = {'xtol':1e-6} in the next line
    sol = optimize.root(residual, guess, method='lm') 
    res = residual(sol.x)
    
    while abs(res).max() > 1e-5:
        print('Tried timestep %g; residual: %g, trying again with (1/3)*timestep' % (delta_t / secperyear, abs(res).max(),)) 
        delta_t = delta_t / 3       
        sol = optimize.root(residual, guess, method='lm') 
        res = residual(sol.x)
    
    y = sol.x
    P, r, L, T, LX, X = np.split(y, 6)
        
    rho = 10.0**rho_interp((np.log10(P),X))
    
    if gamma_av(rho[0],T[0],X[0]) < gamma_melt:
        mn_solid = 0.0
    else:
        gamma = np.zeros_like(rho)
        for i in range(len(gamma)):
            gamma[i] = gamma_av(rho[i], T[i], X[i])
        gamma_interp = interpolate.interp1d(gamma, m/mass)
        mn_solid = mass * gamma_interp(gamma_melt)

    dm_solid = mn_solid - m_solid
    m_solid = mn_solid
    
    max_delta_lnT = np.max(np.abs((T-guess[3*ngrid:4*ngrid])/guess[3*ngrid:4*ngrid]))
    max_delta_lnLX = 0.0
    max_deltaX = np.max(np.abs((X-guess[5*ngrid:6*ngrid])))
    count +=1
    time += delta_t

    # store history
    t_hist = np.append(t_hist, time/secperyear)
    L_hist = np.append(L_hist, L[-1]/Lsun)
    deltat_hist = np.append(deltat_hist, delta_t/secperyear)
    res_hist = np.append(res_hist, abs(res).max())
    model_hist = np.append(model_hist, count)

    # output some information about the step
    print('Step %d, time=%lg Gyr:' % (count,time/secperyear/1e9))
    print('Timestep: %g yrs' % (delta_t/secperyear))
    print('Residual: %g; max lnT change: %g; max X change: %g, max LX change: %g' % (abs(res).max(), max_delta_lnT, max_deltaX, max_delta_lnLX))
    print('R_9= ', r[-1]/1e9)
    print('rho_c,6, T_c,6, P_c,23  = ', rho[0]/1e6, T[0]/1e6, P[0]/1e23)
    print('L = ', L[-1]/Lsun, ' Lsun')
    print('m_solid/mass = ', m_solid/mass)
    print('---------------------------------------------------')

    # rescale the timestep to try to get a fixed fractional change in T
    delta_t1 = 1e99
    delta_t2 = 1e99
    delta_t3 = 1e99 
    delta_t4 = 1e99   
    if max_delta_lnT > 0.0:
        delta_t1 = delta_t * 0.03/max_delta_lnT
    if max_deltaX > 0.0:
        delta_t2 = delta_t * 0.03/max_deltaX
    if dm_solid > 0.0:
        delta_t3 = delta_t * 0.1 * m_solid/dm_solid        
    if max_delta_lnLX > 0.0:
        delta_t4 = delta_t * 0.03/max_delta_lnLX
    #if dm_solid>0.0:
    #delta_t = min(1e6*secperyear,delta_t1,delta_t2,delta_t3)
    #else:
    new_delta_t = min(3e8*secperyear,delta_t1,delta_t2,delta_t3, delta_t4)
    old_delta_t = delta_t
    delta_t = min(new_delta_t, 10.0 * old_delta_t)

    # update the plots
    plot_handle_L.set_xdata(t_hist)
    plot_handle_res.set_xdata(model_hist)
    plot_handle_deltat.set_xdata(model_hist)
    plot_handle_L.set_ydata(L_hist)
    plot_handle_res.set_ydata(res_hist)
    plot_handle_deltat.set_ydata(deltat_hist)
    plot_handle_T.set_ydata(T)
    plot_handle_T.set_xdata(P)
    plot_handle_mL.set_ydata(L/Lsun)
    
    
    ind = LX<0.0
    plot_handle_mLX.set_ydata(abs(LX[ind]))
    plot_handle_mLX.set_xdata(m[ind]/Msun)
    ind = LX>0.0
    plot_handle_mLX2.set_ydata(abs(LX[ind]))
    plot_handle_mLX2.set_xdata(m[ind]/Msun)
    
    plot_handle_mr.set_ydata(r)
    plot_handle_rrho.set_ydata(rho)
    plot_handle_rrho.set_xdata(r)        
    plot_handle_X.set_ydata(X)        
    plot_handle_solid.set_xdata((m_solid/Msun,m_solid/Msun))        
    plot_handle_solid2.set_xdata((m_solid/Msun,m_solid/Msun))        
    plot_handle_solid3.set_xdata((m_solid/Msun,m_solid/Msun))        
    fig.suptitle(r'$\mathrm{Model}\ %d\ \ \ t=%g\ \mathrm{Gyr}\ \ \ L=%.2e\ L_\odot$' % (count, time/secperyear/1e9, L[-1]/Lsun))
    plt.savefig('png/wdevol%05d.png' % (count,), dpi=200)
    ax_res.relim()
    ax_res.autoscale_view()
    ax_deltat.relim()
    ax_deltat.autoscale_view()
    #ax_LX.relim()
    #ax_LX.autoscale_view()
    fig.canvas.draw()
    plt.pause(1e-5)

print('End -- press enter to quit')
input()

print('Writing history.dat file...')
np.savetxt('history.dat', np.c_[t_hist,L_hist])


