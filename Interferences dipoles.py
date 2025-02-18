import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as ct
from tqdm import tqdm
from scipy.optimize import bisect

eps0 = ct.epsilon_0
mu0 = ct.mu_0
pi = ct.pi
c = ct.c

list_x0_dipoles=np.array([0,0])   # x-positions of the middles of the dipoles in microns (dipoles along x)
list_y0_dipoles=np.array([-20,0])  # y-positions of the middle of the dipoles in microns (dipoles along x)
list_moment_dipole = [1e-26,0.5e-26] # list of the dipole moments (p=d*q) 

lambd = 7.2*1e-6 # wavelength 
q=1*1.6e-19   # individual charge of one of the moving charges (defaut /q_electron/)
omega = 2*pi*c/lambd
tm = 2*pi/omega
list_t= [0,tm/4]

x_max, y_max = 100, 100 # in microns, plot the field between -x_max to + x_max, -y_max to + y_max
XX,YY = np.meshgrid(np.arange(-x_max,x_max+1,1),np.arange(-y_max,y_max+1,1))
XX = XX* 1e-6
YY = YY* 1e-6
list_x0_dipoles = list_x0_dipoles* 1e-6
list_y0_dipoles = list_y0_dipoles* 1e-6

def E(x,y,x0,y0,q,d,t,omega):
    
    # x1 = x0 + d/2*cos(wt) x2 = x0 -d/2*cos(wt) v1 = -d/2 w sin(wt) v2 = d/2 w sin(wt) a1 = -d/2 w**2 cos(wt) a2 = d/2 w**2 cos(wt)
    
    def f1(tr1):   # equal 0 when tr1 is the retarded time of interaction at t in x, y from charge 1
        return(((x-(x0+d/2*np.cos(omega*tr1)))**2+(y-y0)**2)**(1/2) - c*(t-tr1))
    def f2(tr2):   # equal 0 when tr2 is the retarded time of interaction at t in x, y from charge 2
        return(((x-(x0-d/2*np.cos(omega*tr2)))**2+(y-y0)**2)**(1/2) - c*(t-tr2))
    
    dt_max = max(x_max,y_max)*1e-6*2/c  # maximum retarded time given the ploting grid
    tr1 , tr2 = bisect(f1,t,t-dt_max,xtol=1e-17), bisect(f2,t,t-dt_max,xtol=1e-17)
    xr1,xr2 = x0 +d/2 *np.cos(omega*tr1), x0 -d/2*np.cos(omega*tr2) # x-positions of the charges at respective retarded times tr1 and tr2
    r1x, r2x ,r1y , r2y = x-xr1, x-xr2 , y- y0, y-y0
    R1 , R2 = (r1x**2+r1y**2)**(1/2) , (r2x**2+r2y**2)**(1/2) # retarded r-vector as defined in Griffiths
    v1r, v2r = -d/2*omega*np.sin(omega*tr1), d/2*omega*np.sin(omega*tr2) # x-velocities of the charges
    a1r, a2r = -d/2*omega**2*np.cos(omega*tr1), d/2*omega**2*np.cos(omega*tr2) # x-accelerations of the charges
    
    u1x, u2x = c*r1x/R1 - (v1r) , c*r2x/R2 - (v2r)  # u = c *(^r) - v  as defined in Griffiths
    u1y, u2y = c*r1y/R1  , c*r2y/R2
    u1 , u2 = (u1x**2 + u1y**2)**(1/2) , (u2x**2 + u2y**2)**(1/2)

    Ex1_1 = q / (4*pi*eps0)*R1/((r1x*u1x + r1y*u1y)**3) * ((c**2-v1r**2)*u1x)  # derivation of each component of the fields from Liénard-Wiechert Potentials 
    Ex2_1 = -q / (4*pi*eps0)*R2/((r2x*u2x + r2y*u2y)**3) * ((c**2-v2r**2)*u2x)
    Ex1_2 = q / (4*pi*eps0)*R1/((r1x*u1x + r1y*u1y)**3) * -r1y*u1y*a1r
    Ex2_2 = -q / (4*pi*eps0)*R2/((r2x*u2x + r2y*u2y)**3) * -r2y*u2y*a2r
    
    Ey1_1 = q / (4*pi*eps0)*R1/((r1x*u1x + r1y*u1y)**3) * ((c**2-v1r**2)*u1y)
    Ey2_1 = -q / (4*pi*eps0)*R2/((r2x*u2x + r2y*u2y)**3) * ((c**2-v2r**2)*u2y)
    Ey1_2 = q / (4*pi*eps0)*R1/((r1x*u1x + r1y*u1y)**3) * r1x*u1y*a1r
    Ey2_2 = -q / (4*pi*eps0)*R2/((r2x*u2x + r2y*u2y)**3) * r2x*u2y*a2r
    
    Ex = Ex1_1 + Ex1_2 + Ex2_1 + Ex2_2
    Ey = Ey1_1 + Ey1_2 + Ey2_1 + Ey2_2
    return(Ex,Ey)

list_Ex , list_Ey = [], []

for ti in tqdm(list_t,position=0, leave=True):
    ni, nj = np.shape(XX)
    Ex, Ey = np.zeros_like(XX), np.zeros_like(XX)
    for i in tqdm(range(ni),position=1, leave=False):
        for j in range(nj):
            x, y  = XX[i,j], YY[i,j]
            E1 = E(x,y,list_x0_dipoles[0],list_y0_dipoles[0],q,list_moment_dipole[0]/q,ti,omega)  # field from dipole 1
            E2 = E(x,y,list_x0_dipoles[1],list_y0_dipoles[1],q,list_moment_dipole[1]/q,ti,omega)  # field from dipole 2
            Ex[i,j]=E1[0] + E2[0]
            Ey[i,j]=E1[1] + E2[1]
    list_Ex.append(Ex)
    list_Ey.append(Ey)
    

Ex1 = list_Ex[0].copy()
Ey1 = list_Ey[0].copy()
Ex2 = list_Ex[1].copy()
Ey2 = list_Ey[1].copy()
for i in range(len(list_x0_dipoles)):  # apply masks onto the dipole position (r=0 gives error)
    idx_x = np.argmin(np.abs(XX[0]-list_x0_dipoles[i]))
    idx_y = np.argmin(np.abs(YY[:,0]-list_y0_dipoles[i]))

    Ex1[idx_y-1:idx_y+2,idx_x-1:idx_x+2]=np.zeros((3,3))
    Ey1[idx_y-1:idx_y+2,idx_x-1:idx_x+2]=np.zeros((3,3))

    Ex2[idx_y-1:idx_y+2,idx_x-1:idx_x+2]=np.zeros((3,3))
    Ey2[idx_y-1:idx_y+2,idx_x-1:idx_x+2]=np.zeros((3,3))

A = ((Ex1**2+Ey1**2)+(Ex2**2+Ey2**2))  # Amplitude of the field

plt.figure()
plt.pcolor(XX*1e6,YY*1e6,A**(0.5))
plt.xlabel(r'$x (\mu m)$')
plt.ylabel(r'$y (\mu m)$')
cb = plt.colorbar()
cb.set_label(r'$I_{scatt}$')
plt.contour(XX*1e6,YY*1e6,A, levels=[1],colors=['red'])
plt.show()