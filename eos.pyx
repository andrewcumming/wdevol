#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import cython
from scipy import optimize
import numpy as np

cdef double Ye(double X):
    return 0.5

cdef double Yi(double X):
    return X/12 + (1-X)/16

cdef double YZ2(double X):
    return 36*X/12 + 64*(1-X)/16

def gamma(double rho, double T, double X):
    # calculate dln P/dln rho numerically
    eps = 1e-3
    P = pressure(rho, T, X)
    rho1 = rho * (1.0 + eps)
    P1 = pressure(rho1, T, X)
    return np.log(P1/P) / eps

def pressure(double rho, double T, double X):
    cdef double myYe, Knr, Kr, Pnr, Pr, Pe, Pc, Pi
    cdef rY43
    if rho < 0: return 0.0
    # composition
    myYe = Ye(X) 
    # degenerate electrons
    rY43 = (myYe*rho)**(4.0/3.0)
    Pnr = 9.92e12 * (myYe*rho)**(5.0/3.0)
    Pr = 1.23e15 * rY43
    # use the formula from Paczynski to interpolate
    Pe = 1.0 / ((1.0/Pnr)**2 + (1.0/Pr)**2)**0.5
    # Coulomb pressure
    Pc = - 8.7e12 * rY43
    # ion thermal pressure
    Pi = Yi(X)*rho*T*1.38e-16/1.67e-24
    return Pe + Pc + Pi

def heat_capacity(double rho, double T, double X):
    return 2.48e8 * Yi(X)
    
def thermal_conductivity(double rho, double T, double X):
    # assume x<<1 and also Coulomb log =1
    return 2.33e3 * rho * T * Ye(X)**2/YZ2(X)
    
def find_rho_eqn(double rho_guess, double pressure_to_find, double T, double X):
    return pressure_to_find - pressure(rho_guess, T, X)

def find_rho(double P, double T, double X):
    sol = optimize.brentq(find_rho_eqn,1e-6,1e10,args=(P,T,X))
    return sol

def gamma_av(double rho, double T, double X):
    cdef double Z53
    Z53 = 1.650963 * X + 2*(1-X)
    #Z53 = X*6**(5.0/3.0)/12 + (1-X)*8**(5.0/3.0)/16
    Z53 = Z53 / Yi(X)
    return 2.27e5*rho**(1.0/3.0)/T * Z53 * Ye(X)**(1.0/3.0)

#def Ltop(double T):
#    return 2e6 * T**3.5  # Using eq. 4.1.12 for Mestel cooling law from Shapiro & Teukolsky, drop mass factor
#    # Mochkovitch (1983) has a formula from van Horn (1968)
#    #return 1e-5*Lsun*(M/Msun)*(T/1e6)**2.7

#def Ttop(double L):
#    return (L/2e6)**(2.0/7.0)


