#THIS CODE SOLVES THE EQUATIONS GIVEN BY KHERANI ET AL (2012, DOI: 10.1111/j.1365-246X.2012.05617.x)..
#(1) WAVE EQUATION FOR AMPLITUDE (W):
#d2w/dt2=(1/rho)grad(1.4 pn div.W)-((grad pn)/rho)div.(rho W)+(1/rho)grad(W.div)p+d((mu/rho)div.div (W)/dt-d(W.div W)/dt
#(2) DENSITY EQUATION (rho)
#d rho/dt+div.(rho W)=0
#(3) ENERGY OR PRESSURE EQUATION (pn)
#d pn/dt+div. (pn W)+(1.4-1)div. W=0
#THIS NUMERICAL CODE HAS SECOND-ORDER ERROR IN SPACE..
#THEREFORE, THERE IS A POSSIBILITY TO IMPROVE...
#ALSO, THIS CODE USES GAUSS-SEIDEL METHOD TO SOLVE THE MATRIX EQUATION.
#THIS METHOD IS SUBJECTIVE..
#THE CODE CAN REPRODUCE OBSERVATIONS UP TO 70%-80% QUALITATIVELY..

#THE CODE USES MKS UNITS AND SOLVES THE EQUATIONS ON THE (X-Y) PLANE 
#WHICH REPRESENTS (LONGITUDE-ALTITUDE) OR (LATITUDE-ALTITUDE)
#THIS CODE USES GROUND FORCING (Y=0 KM) TO EXCITE THE WAVES
#THIS FORCING VARIES IN TIME AND X AS A GAUSSIAN
#FORCING IS MECHANICAL IN NATURE, I.E., VERTICAL WIND FORM
#THE CODE WORKS WELL WITH dt=dy AND dy<=dx<=2.*dy AND 5KM<=dy<=10 KM
#LONGITUDE OR LATITUDE BOUNDARIES SHOULD BE FURTHER FROM THE FORCING LOCATION TO AVOID WAVE REFLECTIONS..
#THE UPPER BOUNDARY (YMAX) SHOULD BE EQUAL OR GREATER THAN 400 KM.
#=============================================================================#
#sigma_t is the temporal Gaussian packet thickness
#t_o is the time when forcing reaches maximum amplitude and should be greater than 2*sigma_t
#t_f is the final simulation time and should be more than 2*t_o
#=============================================================================#

#%%
#==================================MAIN====================================== 
#(wx,wy) are AGWs amplitudes in (x,y) directions, i.e., Longitudinal and transverse 
#(rho_o,tn_o,pn_o) are atmospheric density, temperature, and pressure
#(wx_m,wy_m)=(wx(t-dt,x,y),wy(t-dt,x,y))
#(wx_o,wy_o)=(wx(t,x,y),wy(t,x,y)
#(rho_o,tn_o,pn_o)=(rho(t-dt,x,y),tn(t-dt,x,y),pn(t-dt,x,y))

#%%
#==============================================================================
#=Simulation (X-Y) plane represents the plane where
#(+X,+Y) represent west OR north and vertical upwards (altitude) 
#respectively.

import scipy

from pylab import *
from numpy import *
#from pyiri2016 import *
from nrlmsise_2000 import *
import iri2016
from scipy import *
from scipy.ndimage import *
from scipy.special import erf
from numpy import trapz
from signal_alam import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

import time
start = time.time()

matplotlib.rc("mathtext",fontset="cm")        #computer modern font 
matplotlib.rc("font",family="serif",size=12)

gma=1.33
def d1_3(n2,n3,data):
    return repeat(repeat(data[newaxis,:],n3,axis=0)[newaxis,:,:],n2,axis=0)

def d1_2(n,data):
    return repeat(data[newaxis,:],n,axis=0)

def d1_23(n,data):
    return repeat(data[:,:,newaxis],n,axis=2)

#=============================================================================#
def pdf(x):
    return exp(-x**2/2)

def cdf(x):
    return (1 + erf(x/sqrt(2)))

def skew(x,mu,sig,a):
    t=(x-mu) / sig
    return pdf(t)*cdf(a*t)
#=============================================================================#
def ddf(x,mu,sig):
    val = np.zeros_like(x)
    val[(-(1/(2*sig))<=x-mu) & (x-mu<=(1/(2*sig)))] = 1
    return val

#=============================================================================#
def div_f(f0,f1):
    return gradient(f0)[0]/dx_m+gradient(f1)[1]/dy_m 

#%%
def sum_gr(ndim,ndata,data):
    data_n=0*data
    
    if ndim==0:
        for j in range (ndata):
            if j==0:
                data_n[j,:]=(data[j+1,:]+data[j,:])/2.
            elif j==ndata-1:
                data_n[j,:]=(data[j-1,:]+data[j,:])/2.
            else:
                data_n[j,:]=data[j+1,:]+data[j-1,:]
    
    if ndim==1:
        for j in range (ndata):
            if j==0:
                data_n[:,j]=(data[:,j+1]+data[:,j])/2.
            elif j==ndata-1:
                data_n[:,j]=(data[:,j-1]+data[:,j])/2.
            else:
                data_n[:,j]=data[:,j+1]+data[:,j-1]
    return data_n

#%%
def data_antes(dim,ndim,data):
    data_n=0*data
    if dim==1:
        data_n=0*data
        data_n[1:-1]=data[0:-2]
        data_n[0]=data_n[1];data_n[-1]=data_n[-2];
    else:
        if ndim==0:
            data_n[1:-1,:]=data[0:-2,:]
            data_n[0,:]=data_n[1,:];data_n[-1,:]=data_n[-2,:];
        if ndim==1:
            data_n[:,1:-1]=data[:,0:-2]
            data_n[:,0]=data_n[:,1];data_n[:,-1]=data_n[:,-2];
    return data_n

#%%
def data_proximo(dim,ndim,data):
    data_n=0*data
    if dim==1:
        data_n=0*data
        data_n[1:-1]=data[2:]
        data_n[0]=data_n[1];data_n[-1]=data_n[-2];
    else:
        if ndim==0:
            data_n[1:-1,:]=data[2:,:]
            data_n[0,:]=data_n[1,:];data_n[-1,:]=data_n[-2,:];

        if ndim==1:
            data_n[:,1:-1]=data[:,2:]
            data_n[:,0]=data_n[:,1];data_n[:,-1]=data_n[:,-2];
    return data_n

#%%
def ambiente_atmos(iw,x2,y2):
    global rho_amb,tn_amb,r_g,nu_nn,lambda_c
    global pn, sn
    if iw==0:
        lat_ep=38;lon_ep=143
        year,month,dom=2011,3,9
                
        d0 = datetime.date(year,1,1)
        d1 = datetime.date(year, month, dom)
        delta = d1 - d0
        doy=delta.days 
        ut=3;lt=ut+lon_ep/15.
        
        f107A,f107,ap=150,150,4 #300,300,400
        
        f=nrl_msis(doy,ut*3600.,lt,f107A,f107,ap,lat_ep,lon_ep,dy,y[0],ny)
        #f=msise_model(doy,ut*3600.,lt,f107A,f107,ap,lat_ep,lon_ep,dy,y[0],ny)
        tn_msis=f[1];#tn_msis=0*tn_msis+tn_msis.mean()
        den_ox=f[2]*1.e+06;den_n=f[3]*1.e+06;den_o2=f[4]*1.e+06;den_n2=f[5]*1.e+06;
        n_msis=den_ox+den_n+den_o2+den_n2;
        rho_msis=f[6]*1.e+03
        mean_mass=rho_msis/n_msis
        
        b_c=1.38e-23; 
        rg_msis=b_c/mean_mass;
        pn_msis=rg_msis*rho_msis*tn_msis;
        sn_msis=sqrt(gma*pn_msis/rho_msis)
        
        nu_msis=pi*(7*5.6e-11)**2.*sn_msis*n_msis    
        visc_mu_1=3.563e-07*tn_msis**(0.71);
        visc_mu_2=1.3*pn_msis/nu_msis;
        lambda_msis=sn_msis**2./nu_msis                                                       #Conductividade termica
        
        # rho_msis=0*rho_msis+rho_msis[:20].mean()
        # pn_msis=0*pn_msis+pn_msis[:20].mean()
        # tn_msis=0*tn_msis+tn_msis[:20].mean()
        # rg_msis=0*rg_msis+rg_msis[:20].mean()
        sn_msis=sqrt(gma*pn_msis/rho_msis)
        
        rho_amb=d1_2(nx,rho_msis)
        tn_amb=d1_2(nx,tn_msis)
        sn=d1_2(nx,sn_msis)
        r_g=d1_2(nx,rg_msis)
        nu_nn=d1_2(nx,nu_msis)
        lambda_c=d1_2(nx,lambda_msis)
        
    if iw==1:
        a=0.25e+25*32*1.6e-27*1.e-08#*1.e+06; 
        c=-4.5;d=-19.5;c=-3.5
        yr=130.;r_y=y2/yr-1
        rho_amb=a*(exp(c*r_y)+exp(d*r_y))                                             #Densidade de massa (kg/m³)
        
        a1=310;b1=200;c1=-1.;d1=0.5
        a2=220;b2=250;c2=-0.9;d2=6.
        a3=220;b3=135;c3=-0.2;d3=7.
        a4=120;b4=250;c4=-1.5;d4=18.
        a5=220;b5=220;c5=-0.25;d5=5.
        y_1=10.;y_2=65.;y_3=130.;y_4=100.;y_5=230.;
        
        r_y1=y2/y_1-1;r_y2=y2/y_2-1;r_y3=y2/y_3-1
        r_y4=y2/y_4-1;r_y5=y2/y_5-1
        
        first=a1-b1/(exp(c1*r_y1)+exp(d1*r_y1))
        second=-a2+b2/(exp(c2*r_y2)+exp(d2*r_y2))
        third= a3-b3/(exp(c3*r_y3)+exp(d3*r_y3))
        fourth=-a4+b4/(exp(c4*r_y4)+exp(d4*r_y4))
        fifth=a5-b5/(exp(c5*r_y5)+exp(d5*r_y5))
        sixth=-720+1.25*(first+second+0*fourth)+8.5*third+2.*fifth 
        tn_amb=0.5*sixth                                                              #Temperatura atmosferica (K)
    
        r_g=150.*(1.+sqrt(y2*1.e-03+5.)/5.);                                        #constante Boltzman/massa
        
        b_c=1.38e-23;                                                               #Constante Boltzmann
        mean_mass=b_c/r_g                                                           #massa atmosferica
        nn=rho_amb/mean_mass                                                          #Densidade numerica
        pn=r_g*rho_amb*tn_amb                                                           #Pressão atmosferica
        sn=sqrt(1.33*pn/rho_amb)                                                       #velocidade de som
        
        nu_nn=pi*(7*5.6e-11)**2.*sn*nn                                              #frequencia da colisão
        lambda_c=sn**2./nu_nn                                                       #Conductividade termica
        
    return

#%%
def atmos_evolve(rho_o,tn_o,pn_o,wx,wy):
    div_w=div_f(wx,wy)
    div_flux=div_f(rho_o*wx,rho_o*wy)
    # div_flux_x=wx*gradient(rho_o)[0]/gradient(x2_m)[0]
    # div_flux_y=wy*gradient(rho_o)[1]/gradient(y2_m)[1]
    # div_flux=div_flux_x+div_flux_y
    rho=rho_o-0.5*dt*div_flux
    
    div_flux=div_f(tn_o*wx,tn_o*wy)
    # div_flux_x=wx*gradient(tn_o)[0]/gradient(x2_m)[0]
    # div_flux_y=wy*gradient(tn_o)[1]/gradient(y2_m)[1]
    # div_flux=div_flux_x+div_flux_y        
    
    #gma=1.33#rho_ho/rho
    tn=tn_o-0.5*dt*(div_flux+(gma-1.)*tn_o*div_w)
    
    div_flux=div_f(pn_o*wx,pn_o*wy)        
    # div_flux_x=wx*gradient(pn_o)[0]/gradient(x2_m)[0]
    # div_flux_y=wy*gradient(pn_o)[1]/gradient(y2_m)[1]
    # div_flux=div_flux_x+div_flux_y
    
    pn=pn_o-0.5*dt*(div_flux+(gma-1.)*pn_o*div_w)/1.
    
    return (rho,tn,pn)

#%%
def ambiente_iono(iw,x,yi):
    global no_y,n_o,nu_in,nu_0,gyro_i,b_o
    if iw==0:
        import iri2016.profile as iri
        from datetime import datetime, timedelta
        #yi=y2[0,i_ion:]
        time_start_stop = (datetime(2011, 3, 9, 0,42,0), datetime(2011, 3, 9,4,42,0))
        time_step = timedelta(minutes=5)
        alt_km_range = (yi[0], yi[-1], dy)
        glat = 38
        glon = 143
        
        data_iri=iri.IRI(datetime(2011,3,9,3,0,0), alt_km_range, glat, glon)
        ne_iri=data_iri.ne
        sim = iri.timeprofile(time_start_stop, time_step, alt_km_range, glat, glon)

        ax = figure().gca()
        ax.contourf(sim.time, sim.alt_km, sim.ne.T, shading="nearest")
        ax.set_title("Number Density")
        ax.set_xlabel("time [UTC]")
        ax.set_ylabel("altitude [km]")
        

        [n_o,x3]=meshgrid(ne_iri,x)#d1_2(nx,ne_iri)
        sc_h=30.
        nu_in=1.e+03*exp(-(yi-80.)/sc_h)                                            #A perfile (em altitude) da frequencia 
                                                                                    #da colisão (nu_in)..
    #    nu_0=nu_in
    #    i_500=abs(y-500.).argmin()
    #    for i in range (i_500,ny):
    #        nu_0[:,i]=nu_0[:,i_500]
                                                                            
        b_o=30.e-06                                                                 #o campo geomagnetico em Tesla
        q_c=1.6e-19;m_i=1.67e-27;z_i=16.
        gyro_i=q_c*b_o/(z_i*m_i)                                                    # a freuqnecia de giração da ions
    
        gyro_e=-gyro_i*1837.
        show()
    #    m_p=1.67e-27;
    #    pn_msis=b_c*n_msis*tn_msis
    #    sn_msis=sqrt(1.4*pn_msis/rho_msis)
    #    mass_msis=rho_msis/n_msis
    #    
    #    omg_i=q_charge*mag_o/(mass_msis[n1-n1_i:]);
    #    omg_e=-omg_i*1837.
    #    a=mass_msis[n1-n1_i:]/m_p#16.;#a is neutral mass in amu
    #
    #    nu_ii=0.22*ne_iri*1.e-06/(ti)**(3./2.);
    #    nu_in=2.6e-9*1.e-06*(5*n_msis[n1-n1_i:]+ne_iri)*a**(-1./2.)        
    #    a=2.*sn_msis[n1-n1_i:]*8./(3.*sqrt(pi))
    #    nu_in=1.e+06*a*rho_msis[n1-n1_i:]
    #    nu_in=1.*nu_in+0*nu_ii
    #    
    #    nu_ei=34*ne_iri*1.e-06/(ti)**(3./2.);
    #    nu_en=5.4e-16*n_msis[n1-n1_i:]*sqrt(ti)+0*nu_ei
    #    nu_en=1.*nu_ei+0*nu_en
    if iw==1:
        y2=yi
        np=1.e+12;yp=200.;r=y2/yp
        a=2;b=-10.
        n_of=2.*np/(exp(a*(r-1.))+exp(b*(r-1.)))                                    #Perfile (em altitude (y)) 
        
        np=1.e+11;a=5.;b=-60.
        n_oe=2.*np/(exp(a*(r-0.5))+exp(b*(r-0.5)))
        
        n_ef=5.e-01*np*(r+0.5)**2./5.
        n_o=n_of+n_oe+n_ef
        
        sc_h=30.
        nu_in=1.e+03*exp(-(y2-80.)/sc_h)                                            #A perfile (em altitude) da frequencia 
                                                                                    #da colisão (nu_in)..
    #    nu_0=nu_in
    #    i_500=abs(y-500.).argmin()
    #    for i in range (i_500,ny):
    #        nu_0[:,i]=nu_0[:,i_500]
                                                                            
        b_o=30.e-06                                                                 #o campo geomagnetico em Tesla
        q_c=1.6e-19;m_i=1.67e-27;z_i=16.
        gyro_i=q_c*b_o/(z_i*m_i)                                                    # a freuqnecia de giração da ions
    
        gyro_e=-gyro_i*1837.
        
    return ()   

#%%
def iono_evolve(n_o,vx,vy):
    fx_o=vx*gradient(n_o)[0]/2.        
    fy_o=vy*gradient(n_o)[1]/2.
    n=n_o
    vnx=dx_m/dt;vny=dy_m/dt
    for iter in range (11):                                                     #ITERATION LOOP PARA GAUSS-SEIDEL CONVERGENCE
        fx_g=vx*gradient(n)[0]/2.
        fy_g=vy*gradient(n)[1]/2.
        fx=(fx_o+fx_g)/2.                                                       #SEMI IMPLICIT CRANK-NICOLSON TIME INTEGRATION
        fy=(fy_o+fy_g)/2.
        n=n_o-0.5*(fx/vnx+fy/vny)                                                   #EQUAÇÃO NUMERICA DA DENSIDADE
#        n[:,0]=n[:,1];n[:,-1]=n[:,-2];                                          #condições contorno na ALTITUDE 
    return n
#%%
def vel(b_o,nu,gyro,wx,wy):
    global mu_p                                                                 
    # |vx| | mu_p mu_h | |Ex|
    # |  |=|           | |  |
    # |vy| |-mu_h mu_p | |Ey|
    kappa=gyro/nu
    mu_p=kappa/(b_o*(1.+kappa**2.))                                             #PEDERSON MOBILITY
    mu_h=kappa**2./(b_o*(1.+kappa**2.))                                         #HALL MOBILITY
    lat=radians(38)
    mag_m=8.e+15         #Tm^3
    r_ea=6.371e+06
    by=-2.*mag_m*sin(abs(lat))/r_ea**3.;
    bz=mag_m*cos(lat)/r_ea**3.;
    bx=0
    wz=0
    ey=wz*bx-wx*bz
    ex=wy*bz-wz*by
    ez=wx*by-wy*bx
#    ex=wy*b_o*cos(lat);ey=-wx*b_o*cos(lat)
    vx=mu_p*ex+mu_h*ey
    vy=mu_p*ey-mu_h*ex
    return (vx,vy)
#%%
def agw_dispersion(idim):
    if idim==0:
        delta=2.*dx_m;wkv=wk_x;wkh=wk_y
    if idim==1:
        delta=2.*dy_m;wkv=wk_y;wkh=wk_x
    
    #gma=1.33
    pn=pn_o#r_g*rho_o*tn_o;
    c_s=sqrt(gma*pn/rho_o);#c_s=0*c_s+c_s[:,:20].mean()     # Sound speed profile
    gr_pn=gradient(pn)[idim];dy_m2=2.*dy_m
    zeta=(1./rho_o)*gr_pn/delta
    k0=zeta/c_s**2.
    k0_antes=roll(k0,-1,axis=idim)#data_antes(2,idim,k0)
    k0_proximo=roll(k0,1,axis=idim)#data_proximo(2,idim,k0)
    mu=0.5*(k0_proximo+k0_antes)*delta/2.
    mu=cumsum(mu,idim)/2.;#mu=9.*mu/abs(mu).max()
    omega_ac=k0*c_s/1.#1.45*gma*zeta/(2.*c_s)
    omega_c2=omega_ac**2.#c_s**2.*(gma*k0/2)**2.#(gma**2.*k0*c_s)**2./4.;
    omega_b2=((gma-1)*k0**2-(k0/c_s**2.)*gradient(c_s**2.)[idim]/delta)*c_s**2.
    omega_b2[omega_b2<0]=abs(omega_b2).min()
    omega_h2=wkh**2.*c_s**2.
    omega_2=0*omega_b2+wkv**2.*c_s**2.+0*omega_h2+0*omega_c2
    omega_mais=sqrt(omega_2+sqrt(omega_2**2.-0*4.*omega_h2*omega_b2))/sqrt(2)
    omega_menos=sqrt(omega_2-sqrt(omega_2**2.-0*4.*omega_h2*omega_b2))/sqrt(2)

    visc_mu=1.3*pn_amb/nu_nn;visc_ki=visc_mu/rho_amb
    nu_col=visc_ki*(-wkh**2-wkv**2.+k0**2.-gradient(k0)[idim]/delta)
    wx_mais=(omega_mais**2.+omega_h2-omega_2)/(wkv*wkh*c_s**2.)
    wx_menos=(omega_menos**2.+omega_h2-omega_2)/(wkv*wkh*c_s**2.)
    
    gamma_ad=(gma-1)*k0**2.
    gamma_e=(k0/c_s**2.)*gradient(c_s**2.)[idim]/delta
    
    wkv2=(omega_mais**2.-omega_c2)/c_s**2.
    wk_real=0*wkv2;wk_im=0*wkv2
    #wkv2[wkv2>0]=0*wkv2[wkv2>0]
    #wkv2[wkv2<0]=sqrt(abs(wkv2[wkv2<0]))
    
    wk_real[wkv2>0]=sqrt(abs(wkv2[wkv2>0]))
    wk_im[wkv2<0]=sqrt(abs(wkv2[wkv2<0]))
    
    return (mu,omega_mais,omega_menos,nu_col,wx_mais,wx_menos,omega_b2,\
            omega_c2,c_s,wk_real,wk_im)#ob2_im,gamma_ad,gamma_e,c_s)

#%%

def agw_propagator_v(omega,wk_y): # vertical
    omega3=repeat(omega[newaxis,:,:],nt,axis=0)
    v_phase=omega3/wk_y;
    if abs(v_phase).min() !=0:
        t_phase=y3_m/v_phase
    else:
        t_phase=0+0*y3_m
        
    wy0_t=0*v_new
    sigma=1./omega
    for it0 in range (nt):
        f_frente=cos(-omega3*t_s[it0]+wk_y*y3_m+0*wk_x*abs(x3_m))
        f_prop=skew(t_s[it0]-t_phase,t3,sigma/1.,0)*f_frente
        wy0_t[it0,:,:]=-(v_new*f_prop).sum(0)
    
        wy0_t[it0,:,0]=v_new[it0,:,0]+0*wy0_t[it0,:,0]

    sigma_x=2.*dx_m*1.e-03;x_o=0.
    f_lon=skew(x3,x_o,sigma_x,0)
    
    wy0=wy0_t*(f_lon)#-f_lon[0,-1,0])
    return wy0

#%%
def agw_propagator_h(omega,wk_x,wy0_x):  # horizontal
    omega3=repeat(omega[newaxis,:,:],nt,axis=0)
    v_phase=omega3/wk_x;
    if abs(v_phase).min() !=0:
        t_phase=abs(x3_m)/v_phase
    else:
        t_phase=0+0*x3_m
        
    wy_h=0*wy0_x
    sigma=1./omega
    for it0 in range (nt):
        f_frente=cos(-omega3*t_s[it0]+wk_x*abs(x3_m))
        f_prop=skew(t_s[it0]-t_phase,t3,sigma/1.,0)*f_frente
        wy_h[it0,:,:]=-(wy0_x*f_prop).sum(0)
        #wy_h[it0,:,0]=0
    return wy_h

#%%
def agw_propagator_resonance(omega,wk_y,wr):
    omega3=repeat(omega[newaxis,:,:],nt,axis=0)
    v_phase=omega3/wk_y;
    if abs(v_phase).min() !=0:
        t_phase=y3_m/v_phase
    else:
        t_phase=0+0*y3_m
        
    wr_prop=0*wr
    sigma=1./omega
    for it0 in range (nt):
        f_frente=cos(-omega3*t_s[it0]+wk_y*y3_m)
        f_prop=skew(t_s[it0]-t_phase,t3,sigma/1.,0)*f_frente
        wr_prop[it0,:,:]=-(wr*f_prop).sum(0)    
    return wr_prop
#%%
data_sism= load('KSN.npy')#DATA_ILLAPEL/Dados_sismico.npy')
t_s=3600*data_sism[0,:]#290000+20000:-270000:800];#segundos
vel_s=1.e-02*data_sism[1,:]#290000+20000:-270000:800]#m/s
i_ep=abs(abs(vel_s)-abs(vel_s).max()).argmin()
i_start=abs(t_s-(t_s[i_ep]-300)).argmin()
i_last=abs(t_s-(t_s[i_ep]+1500)).argmin()
t_s=t_s[i_start:i_last:150];vel_s=vel_s[i_start:i_last:150]
print (t_s[0]/3600,t_s[-1]/3600,len(t_s))
t_ss=t_s;t_s=t_s-t_s[0];nt=len(t_s)
dt=t_s[1]-t_s[0]

figure(11)
plot(t_s, vel_s)

dy=10;dx=dy;                                                            #A resoluções espaciais no kilometros
y=arange(0,300+dy,dy);ny=len(y)                                           #A faixa de altitude
x=arange(0,200+dx,dx);nx=len(x)                                        #A faixa de longitude
i_alt=abs(y-80).argmin()
[y2,x2]=meshgrid(y,x)
y2_m=y2*1.e+03;x2_m=x2*1.e+03
dy_m=dy*1.e+03;dx_m=dx*1.e+03                                                 #As resoluções espaciais no metros

[x3,v3,y3]=meshgrid(x,vel_s,y)
[x3,t3,y3]=meshgrid(x,t_s,y)
x3_m=x3*1.e+03;y3_m=y3*1.e+03

def fonte_rayleigh():
    v_new=0*v3
    v0=3.e+00
    t_phase=(abs(x3)-0)/v0;sigma=dt/2.
    f_prop=skew(t3,t_phase,sigma,0)
    for k in range (ny):
        for j in range (nx):
            v_new[:,j,k]=-convolve(f_prop[:,j,k],v3[:,j,k])[:nt]
    # for it0 in range (nt):
    #     f_prop=skew(t_s[it0]-t_phase,t3,sigma,0)
    #     v_new[it0,:,:]=(v3*f_prop).sum(0)
    return v_new
v_new=fonte_rayleigh()

#%%==================================MAIN====================================== 
                                                   
global rho_o,tn_o,pn_o,rho_amb,tn_amb,pn_amb
global wk_x,wk_y

#%%
time=[];wx3=[];wy3=[];n3=[];data_arrival=[]
wave_all=[];data_amb=[];pr3=[];rho3=[];tn3=[];cs3=[]
o_br=[];omega_all=[];gr_ci3=[]

#%%============================================================================

f=ambiente_atmos(0,x2,y2)
f=ambiente_iono(0,x,y[i_alt:])     
#data_amb.append((rho_amb[0,:],sn[0,:]))

#%%============================INTIALIZATION===================================
pn_amb=r_g*rho_amb*tn_amb
rho_o=rho_amb;tn_o=tn_amb;pn_o=pn_amb
rho_to=0*rho_o;rho_ho=rho_amb
                                                          
#%%SOLUCAO ANALYTICA

i_pl=1

wy_ana=zeros((nt,nx,ny));wx_ana=zeros((nt,nx,ny));
wy=zeros((nt,nx,ny));wx=zeros((nt,nx,ny));

lambda_y0=arange(2*dy_m,ny*dy_m/2,2*dy_m)
for ik in range (len(lambda_y0)):
    lambda_y=lambda_y0[ik]
    lambda_x0=arange(4*lambda_y,6*lambda_y,lambda_y)#
    #lambda_x0=arange(2*lambda_y,max(nx*dx_m/1.,ny*dy_m/2.),2*dx_m)
    for ikx in range (len(lambda_x0)):
        lambda_x=lambda_x0[ikx]#max(sigma_x,2*ikx*dx_m)
        wk_x=2.*pi/lambda_x;wk_y=2.*pi/lambda_y    
        
        f=agw_dispersion(1) # call the function
        muy=f[0];omega_mais=f[1];omega_menos=f[2];nu_col=f[3];
        wx_mais=f[4];wx_menos=f[5]
        omega_br=sqrt(f[6]);omega_ac=sqrt(f[7]);
        #omega_ci=sqrt(abs(f[8]));gamma_ad=f[9];gamma_e=f[10]
        c_s=f[8];
        wk_real=f[9]
        wk_im=f[10]
        print (wk_real.max(),wk_im.max())
        
        f=agw_dispersion(0);
        mux=f[0];
        omega_mais=omega_mais#(omega_mais+f[1])/2.;
        omega_menos=(omega_menos+f[2])/2.
        
#        if ik==0 and ikx==ik and omega_ci.any()!=0:
#            print ('CONVECTIVELY UNSTABLE GWs')                
        wy_alt=exp(-muy)
        wx_alt=exp(-mux)
        w_alt=wx_alt*wy_alt*exp(wk_im*dy_m)
        w_alt3=repeat(w_alt[newaxis,:,:],nt,axis=0)
        nu_col3=repeat(nu_col[newaxis,:,:],nt,axis=0)
        lambda_c3=repeat(lambda_c[newaxis,:,:],nt,axis=0)
        n=1
        wy_damp=exp(2.*nu_col3*t3/(2.*n))*exp(-lambda_c3*t3*wk_y**2.)
        wy_growth=exp(0)#epx(omega_ci*t_s[-1]/(2.*pi))
        omega_aw=(1./n)*sqrt(1.-(n-1)**2./(4.*omega_mais.max()*t_s[-1])**2.)*omega_mais
        omega_gw=(1./n)*sqrt(1.-(n-1)**2./(4.*omega_menos.max()*t_s[-1])**2.)*omega_menos

        sec_mhz=1.e+03/(2.*pi)
        if sec_mhz/dt<sec_mhz*omega_mais.max():
            print ('High-frequency cut-off')
            break
        
        # if omega_aw.min()<omega_ac.max():
        #     print ("EVANESCENT WAVEs")
        #     break
            
        # fr_aw=(1.e+03/(2.*pi))*omega_aw    
        # if fr_aw.min()< 4.5 and fr_aw.min()> 3.5:
        #     print ("HIGH FREQUENCY WAVEs")
        #     break
    
        for i_wv in range (1):
            if i_wv==0: 
                omega=omega_aw
                wx_amp=wx_mais/1.
            if i_wv==1: 
                omega=omega_gw
                wx_amp=wx_menos/1
            
            wy0=agw_propagator_v(omega,wk_y);
            wy_ground=wy0[:,:,0]
            wy_ondas=wy0*w_alt*wy_damp*wy_growth
            
            # w3_res=repeat(exp(wk_im*dy_m)[newaxis,:,:],nt,axis=0)
            # w_prop=agw_propagator_resonance(omega,wk_real,w3_res);
            # wy_resonance=w_prop#*w_alt*wy_damp*wy_growth
            
            # wy_ondas=(wy_ondas+wy_ondas*w_prop)/10.

            wy0_x=(repeat(wy_ondas.max(1)[:,newaxis,:],nx,axis=1)+wy_ondas)/2.    
            wy_x=agw_propagator_h(omega,wk_x,wy0_x)
            
            wy_ana=wy_ana+(wy_ondas+wy_x) # equation (4) y-component
            wy_ana[:,:,0]=wy_ground
            wx_ana=wx_ana+wx_amp*(wy_ondas+wy_x)#gradient(wy_ana)[1] # x-component
            
        wave_all.append([wk_x,wk_y,omega_aw[0,:],omega_gw[0,:],\
                         omega_br[0,:],omega_ac[0,:]])
    print (sec_mhz/dt,sec_mhz*omega_mais.max(),(2.*pi/wk_y).max(),(omega_mais/wk_y).max())            

for it0 in range (nt): # continuity equation start here
    f=atmos_evolve(rho_o,tn_o,pn_o,wx_ana[it0,:,:],wy_ana[it0,:,:])
    rho=f[0];tn=f[1];pn=f[2] # ends here
    
    f=vel(b_o,nu_in,gyro_i,wx_ana[it0,:,i_alt:],wy_ana[it0,:,i_alt:])
    vx=f[0];vy=f[1]

    n=iono_evolve(n_o,vx,vy)
    
    itm=it0-1
    if it0==0:itm=it0
    itp=it0+1
    if it0==nt-1:itp=it0
    
    wy[:,:,0]=wy_ana[:,:,0]
    wx_0=2.*wx[it0,:,:]-wx[itm,:,:];
    wy_0=2.*wy[it0,:,:]-wy[itm,:,:];
    cs2=1.33*pn_o/rho_o#cs2=cs2.mean()+0*cs2
    
    delta_x=gradient(x2_m)[0];delta_y=gradient(y2_m)[1]
    div_w=gradient(wx[it0,:,:])[0]/delta_x+gradient(wy[it0,:,:])[1]/delta_y
    
    
    d2wx=gradient(div_w)[0]/delta_x
    wx[itp,:,:]=(wx_0+cs2*dt**2.*d2wx)
    
    d2wy=gradient(div_w)[1]/delta_y
    wy[itp,:,:]=(wy_0+cs2*dt**2.*d2wy)
    
    wy_num=wy[it0,:,:]
    wx_num=wx[it0,:,:]
    
    f=vel(b_o,nu_in,gyro_i,wx_num[:,i_alt:],wy_num[:,i_alt:])
    vx=f[0];vy=f[1]

    if it0==0:n0_num=n_o
    n_num=iono_evolve(n0_num,vx,vy)
    
    rho_o=rho;tn_o=tn;pn_o=pn
    n_o=n;n0_num=n_num
    
    print ('Time, seconds=',dt,t_s[it0])
    print ('GROUND UPLIFT, m/s=',wy_ana[it0,:,0].max())
    print('AGWs amplitudes, m/s=',round(wy_ana[it0,:,:].max(),2),\
          round(wx_ana[it0,:,:].max(),2))
    print (d2wy.max())

    time.append(t_ss[it0]/60.)
    rho3.append(rho);tn3.append(tn)
    pr3.append(pn);
    wx3.append(wx_ana[it0,:,:])
    wy3.append(wy_ana[it0,:,:])
    #gr_ci3.append(omega_ci)
    cs3.append(c_s)
   
i_xo=abs(x-x.mean()).argmin()
#omega_all.append([omega_ac[i_xo,:],omega_br[i_xo,:],omega_ci[i_xo,:],\
#                  gamma_ad[i_xo,:],gamma_e[i_xo,:]])

print(np.shape(wy3), np.shape(time), np.shape(y))


#%%
save('time.npy',array(time))
save('wy3.npy',array(wy3))
save('wx3.npy',array(wx3))
save('pr3.npy',array(pr3))
save('tn3.npy',array(tn3))
save('cs3.npy',array(cs3))
#save('wave_all.npy',array(wave_all))



import time
end = time.time()
print(end - start)


dados_y = load("wy3.npy")

fig = figure(12)
ax = subplot(111)
im = ax.imshow(100*dados_y[:, 0, :], origin='lower', cmap='seismic')
im.set_interpolation("spline36")
fig.tight_layout()
savefig("agws.png")
show()