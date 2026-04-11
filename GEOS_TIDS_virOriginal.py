from pylab import *
from numpy import *
#matplotlib.use('WxAgg')
#matplotlib.interactive(True)
from mayavi.scripts import mayavi2
from mayavi import mlab
from mayavi.sources.vtk_data_source import VTKDataSource
from tvtk.api import tvtk
from mayavi.modules.api import Outline, GridPlane
from scipy.signal import *
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import *
from mpl_toolkits.basemap import Basemap
from scipy.io import netcdf
import glob, os
from math import degrees as deg, radians as rad 
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

params={'axes.labelweight':'bold'};rcParams.update(params)
font={'family':'serif','weight':'bold','size':14};matplotlib.rc('font', **font);

global i_o,i_amb,i_pot,i_hemi,i_mag,i_inert

i_hemi=float(raw_input('HEMISPHERE, +1 FOR SOUTHERN '))   #+1 for S, -1 for N
i_mag=float(raw_input('MAGNETIC OR SPHERICAL, 0 FOR SPHERICAL '))   #1 for magnetic, 0 for spherical
i_inert=float(raw_input('INERTIA OR COLLISIONAL, 0 FOR COLLISIONAL '))   #1 for INERTIA, 0 for COLLISIONAL
i_amb=float(raw_input('AMBIENT FORCE, 0 FOR NO AMBIENT '))
i_pot=float(raw_input('POLARIZATION POTENTIAL, 1 FOR SOLVING POTENTIAL '))    #
i_o=float(raw_input('PARALLEL DYNAMICS, 1 FOR YES '))

print i_o,i_amb,i_pot

#=============================================================================#

def sim_vol():
    global r_ea
    global np,nf,nq,np_i    #ALTURA, LONGITUDE, LATITUDE
    global r,theta,dr,dtheta,dphi
    global dp_m,df_m,dq_m,dp_i,df_i,dq_i
    global alt,alt_i,lat,lon
    global y_f,z_f,y_sph,z_sph,p,q
    global alt_3,lon_3,lat_3
    
    
    r_ea=6371.;
    
    alt=0;alt_max=0.;i=0
    
    while alt_max <=6000:
        dr=15.
        if alt_max >=600.:dr=60.
        if alt_max >=1200.:dr=120.

        alt=append(alt,alt_max+dr)       
        alt_max=alt.max()
        
    alt_i=alt[10:];np=len(alt);np_i=len(alt_i)
    
    print alt_i
    print np,alt.min(),alt.max()
    

#    dr=15.;alt=arange(0.,6000.,dr);np=len(alt);
#    dr_m=dr*1.e+03;
#    dr_mean=15.   
#    alt_i=arange(75.,6000.,dr);np_i=len(alt_i);
    
    nf=101;dr=15.
    dphi=round(arctan(3.*dr/(r_ea+300.)),5);
    lon_in=-radians(75.)
    lon=arange(lon_in,lon_in+nf*dphi,dphi)
    lon=degrees(lon);
    
    nq=101;dtheta=1.*dphi;
    lat_in=-radians(20.)#-50.*dtheta     
    lat=i_hemi*arange(lat_in,lat_in+nq*dtheta,dtheta)   #SOUTHERN HEMI: i_hemi=1
    lat=degrees(lat);
    
    print len(lat),len(lon)
    
    #=============================================================================#
    
    r=zeros((np,nf,nq));theta=zeros((np,nf,nq));phi=zeros((np,nf,nq))
    
    for i in range (0,np):
        r[i,:,:]=(alt[i]+r_ea)*1.e+03;
    for j in range (0,nf):
        phi[:,j,:]=radians(lon[j])
    for k in range (0,nq):
        theta[:,:,k]=radians(lat[k])
        
    r_o=r/(cos(theta))**2.
    delta=sqrt(1.+i_mag*3.*(sin(theta))**2.)
    q=r_o**3.*sin(theta)/r**2.
    p=r/cos(theta)**2.
    hq=(r/r_o)**3./delta
    hphi=r*cos(theta)
    hp=cos(theta)**3./delta
    
    dp=abs(gradient(p)[0]+abs(gradient(p)[2]))
    dq=abs(gradient(q)[0]+(gradient(q)[2]))
    
    dp_m=dp*hp;dq_m=dq*hq;df_m=dphi*hphi
    x_f=hphi*tan(phi)*1.e-03;y_f=p*hp*1.e-03;z_f=q*hq*1.e-03
    x_sph=x_f;y_sph=y_f*delta;z_sph=z_f*delta
    
    
    alt_3=y_sph-r_ea
    for i in range (0,np):
        for j in range (0,nf):
            for k in range (0,nq):
                if alt_3[i,j,k] < 0:
                    alt_3[i,j,k]=0  
                    
    lat_3=degrees(arctan(z_f/(y_f+0.)))
    lon_3=degrees(arctan(x_f/(y_f+0.)))
    
    print dp_m.min(),dp_m.max(),dq_m.min(),dq_m.max()

    dp_m[:,:,:]=dp_m;dq_m[:,:,:]=df_m.max();df_m[:,:,:]=df_m.max()
    dp_i=dp_m[np-np_i:,:,:];df_i=df_m[np-np_i:,:,:];dq_i=dq_m[np-np_i:,:,:]

    print '======================================================================='
    print 'GRID_SIZES',dp_m.min(),dp_m.max(),dq_m.min(),dq_m.max(),df_m.min(),df_m.max()
    

    fig = figure(1,figsize=(18,12),facecolor='w',edgecolor='k')
    theta_r=180.*arctan2(y_f,z_f)/pi;alt_r=sqrt(y_f**2.+z_f**2.)
    ax=subplot(121)
    pcolor(z_sph[:,0,:],y_sph[:,0,:],y_f[:,0,:]-r_ea);colorbar()   
    plot(z_f[:,0,0:nq:10],y_f[:,0,0:nq:10],color='gray',alpha=0.5)
    labels=around(lat[0:nq:10],decimals=0)
    ii=0
    for i in range (0,nq,10):
        text(z_f[np-1,0,i],y_f[np-1,0,i],str(int(labels[ii])))
        ii=ii+1
    plot(z_f[0:np:10,0,:].T,y_f[0:np:10,0,:].T,color='gray',alpha=0.5)
    labels=around(alt[0:np:10],decimals=0)
    ii=0
    for i in range (0,np,10):
        text(z_f[i,0,int(nq/2.)],y_f[i,0,int(nq/2.)],str(int(labels[ii])))
        ii=ii+1
    gca().add_patch(Circle((0,0),radius=r_ea,fc='b',fill=False,alpha=1))
    gca().add_patch(Circle((0,0),radius=r_ea-200,fc='w',fill=False,alpha=1))    
    
    axis('off')        
    
    ax=subplot(122)
    map= Basemap(projection='ortho',lat_0=lat_3.mean(),lon_0=lon_3.mean(),resolution='l')
    map.etopo(alpha=0.5);
#    map.drawcoastlines()
    map.drawmeridians(arange(lon_3.min(),lon_3.max()+10,10))
    map.drawparallels(arange(lat_3.min(),lat_3.max()+10,10))
    x, y = map(lon_3[0,:,:],lat_3[0,:,:])
    pcolor(x,y,y_f[0,:,:],alpha=0.5)
    
    axins=zoomed_inset_axes(ax, 0.5, loc=1) # zoom = 6
    axins.pcolor(x,y,y_f[0,:,:],alpha=0.5)
#    x1, x2, y1, y2 = lon_3.min(), lon_3.max(), lat_3.min(), lat_3.max()
#    axins.set_xlim(x1, x2)
#    axins.set_ylim(y1, y2)
    xticks(visible=False)
    yticks(visible=False)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    
#    show()

    return()
    
f=sim_vol()

#=============================================================================#

def source_space():
    global iphi_s,iq_s

    ip_s=1
    alt_o=105.;sigma_p=15.
    var_p=exp(-(alt-alt_o)**2./sigma_p**2.)    
    
    iphi_s=int(nf/2.)
    lon_o=lon[iphi_s];sigma_x=1.*5.*degrees(dphi)/2.
    wl=20*degrees(dphi);wk=2.*pi/wl
    var_f=cos(wk*lon)
    var_f=exp(-(lon-lon_o)**2./sigma_x**2.)
    
    iq_s=int(nq/4.)
    lat_o=lat[iq_s];sigma_z=1.*sigma_x
    wl=10.*degrees(dtheta);wk=2.*pi/wl
    var_q=exp(-(lat-lat_o)**2./sigma_z**2.)
    
    source_shape=zeros((np,nf,nq));mask_bound=1+zeros((np,nf,nq))
    
    for i in range (0,np):
        for j in range (0,nf):
            for k in range (0,nq):
#                if alt_3[i,j,k] > 0.:
                source_shape[i,j,k]=var_p[i]*var_f[j]*var_q[k]
                
        
            j_m=int(nf/2.);k_m=int(nq/2.);
            rad_m=sqrt(j_m**2.+k_m**2.)
            rad_o=sqrt((j-j_m)**2.+(k-k_m)**2.)
            if rad_o >= 0.6*rad_m:
                mask_bound[:,j,k]=exp(-(2.*rad_o/rad_m)**2.)
    return (source_shape,mask_bound)

f=source_space()
source_shape=f[0];mask_bound=f[1]
#fig = figure(3,figsize=(18,12),facecolor='w',edgecolor='k')
#contourf(lon,lat,source_shape[8,:,:].T)
#show()

def source_t(t):
    omega=2.*pi/(5.*60.);t_o=2000.;sigma_t=1500.
    s_t=cos(0*omega*t)*exp(-(t-t_o)**2./sigma_t**2.);
    return (s_t)
#=============================================================================#
        
def atmos(alt):
    a=0.25e+25*32*1.6e-27*1.e-08#*1.e+06; 
    c=-4.5;d=-19.5;c=-3.5
    yr=130.;r_y=alt/yr-1
    rho_o=a*(exp(c*r_y)+exp(d*r_y))
    yr=130.;r_y=(alt-dr)/yr-1
    rho_od=a*(exp(c*r_y)+exp(d*r_y))
    yr=130.;r_y=(alt+dr)/yr-1
    rho_ou=a*(exp(c*r_y)+exp(d*r_y))

#    print dr
#    sc_h=15.
#    rho_o=a*exp(-alt/sc_h)
#    rho_od=a*exp(-(alt-dr)/sc_h)
#    rho_ou=a*exp(-(alt+dr)/sc_h)
    
    a1=310;b1=200;c1=-2.;d1=0.5
    a2=220;b2=250;c2=-0.2;d2=10.
    a3=220;b3=160;c3=-0.25;d3=5.5
    a4=120;b4=250;c4=-1.5;d4=18.
    a5=220;b5=220;c5=-0.25;d5=5.
    y1=10.;y2=75.;y3=130.;y4=100.;y5=230.;
    
    r_y1=alt/y1-1;r_y2=alt/y2-1;r_y3=alt/y3-1
    r_y4=alt/y4-1;r_y5=alt/y5-1
    
    first=a1-b1/(exp(c1*r_y1)+exp(d1*r_y1))
    second=-a2+b2/(exp(c2*r_y2)+exp(d2*r_y2))
    third= a3-b3/(exp(c3*r_y3)+exp(d3*r_y3))
    fourth=-a4+b4/(exp(c4*r_y4)+exp(d4*r_y4))
    fifth=a5-b5/(exp(c5*r_y5)+exp(d5*r_y5))
    sixth=-130+1.*(first+second+fourth)+5.2*third++0.5*fifth  
    sixth=-720+1.25*(first+second+fourth)+8.5*third+1.5*fifth 
    temp_o=0.65*sixth
#    fig = figure(1,figsize=(18,12),facecolor='w',edgecolor='k')
#    data=loadtxt('atmos_prof.dat')
#    subplot(121)
#    semilogx(data[:,1],data[:,0])
#    semilogx(rho_o,alt)
#    subplot(122)
#    plot(data[:,2],data[:,0])
#    plot(temp_o,alt)
#    temp_o=data[0:32,2]
    
#    rho_o[40:]=rho_o[40];
#    rho_ou[40:]=rho_ou[40];
#    rho_od[40:]=rho_od[40];
#    temp_o[40:]=temp_o[40];    
    return (rho_o,rho_od,rho_ou,temp_o)


def iono_amb():
    global mag_o, q_charge, omega_i,omega_e,tec_fac, den_amb
    
    q_charge=1.6e-19;b_c=1.38e-23;eps_o=8.854e-12;m_i=1.67e-27;z_i=16.
    mag_o=0.25e-04;omega_i=q_charge*mag_o/(z_i*m_i);r_li=600./omega_i;
    omega_e=-omega_i*1837.
    
    den_100=1.e+18;sc_h=40.;den_neu=den_100*exp(-(alt_i-100.)/sc_h)
    
    yp=3.e+02;enp=2.e+12;enpe=enp*1.e-02*1.e+01
    af=1.;bf=-10;ae=5.;be=-60.
    r_n=(alt_i)/yp;r_of=0.9;r_oe=0.255;r_ov=0.5
    
    den_f=enp/(exp(af*(r_n-r_of))+exp(bf*(r_n-r_of)))
    den_e=enpe/(exp(ae*(r_n-r_oe))+exp(be*(r_n-r_oe)))
    den_ef=5.e-01*enpe*(r_n+r_ov)**2./5.
    den_i=den_f+den_e+den_ef

    
    int_fac=50.*60.*1.e+03
    tec=den_i.sum();tec_fac=int_fac/1.e+18
    tec=5.*tec*tec_fac
    
    t_i=800.;a=16.;#a is neutral mass in amu
    nu_ii=0.22*den_i*1.e-06/(t_i)**(3./2.);
    nu_in_1=(0*nu_ii+2.6e-9*1.e-06*(den_neu+den_i)*a**(-1./2.)) 
    nu_ei=34*den_i*1.e-06/(t_i)**(3./2.);
    nu_en_1=5.4e-16*den_neu*sqrt(t_i)+nu_ei

    w_p1=sqrt(den_i*q_charge**2./(eps_o*m_i*z_i))
    
    den_i[40:]=den_i[40];
    nu_in_1[40:]=nu_in_1[40];
    nu_en_1[40:]=nu_en_1[40];
    
    den_amb=zeros((np_i,nf,nq));nu_in=zeros((np_i,nf,nq));nu_en=zeros((np_i,nf,nq))
    for k in range (0,nq):
        for j in range (0,nf):
            den_amb[:,j,k]=den_i#[np-np_i:,j,k]
            nu_in[:,j,k]=nu_in_1#[np-np_i:,j,k];
            nu_en[:,j,k]=nu_en_1#[np-np_i:,j,k]
            
    return (nu_in,nu_en)

#=============================================================================#
def foto():
#    f=iono_amb()
#        wk_f=2.*pi/(1200.*dphi);wk_q=2.*pi/(200.*dtheta)
#        for j in range (0,nf):
#            for k in range (0,nq):
#                lat_o=13.;sigma=15.*dtheta;gauss_lat=exp(-(abs(-12.+lat[k])-lat_o)**2./sigma**2.);
#                amp=0.5*gauss_lat
#                fac_bg=(1.-0.9*cos(wk_f*(lon[j]-35)))*(1+amp)/1.
#                tec_t[j,k]=f[3]#*fac_bg
#                tec_amb=tec_t
#    foto=zeros((nf,nq))
#    for j in range (0,nf):
#        time=t[i_t]/3600.
#        t_o=-1.5+(lon[nf-1]-lon[j])*1./15.;sigma=1.75;amp_t=exp(-(time-t_o)**2./sigma**2.)
#        period=1.*3600.;fac=-2.*(time-1.5)/sigma**2.;
#        foto[j,:]=0*fac*amp_t/period
    return ()

#=============================================================================#
def mobility():        #(mu_p,-mu_h//mu_h,mu_p)
    
    f=iono_amb()

    nu_in=f[0];nu_en=f[1]
    
    omega=i_inert/dt;nu_eff=nu_in+omega
    kappa=omega_i/nu_eff

    mu_p_i=(kappa/(mag_o*(1+kappa**2.)))
    mu_h_i=(kappa**2./(mag_o*(1+kappa**2.)))
    mu_o_i=(kappa*(1+kappa**2.)/(mag_o*(1+kappa**2.)))
    
    omega=i_inert/dt;nu_eff=nu_en+omega
    kappa=omega_e/nu_eff
    
    mu_p_e=(kappa/(mag_o*(1+kappa**2.)))
    mu_h_e=(kappa**2./(mag_o*(1+kappa**2.)))
    mu_o_e=(kappa*(1+kappa**2.)/(mag_o*(1+kappa**2.)))
    
#    fig = figure(2,figsize=(18,12),facecolor='w',edgecolor='k')
#    semilogx(nu_in[:,0,0],alt_i);semilogx(nu_en[:,0,0],alt_i);legend((r'$\nu_{in}$',r'$\nu_{en}$'))
    
    return (mu_p_i,mu_h_i,mu_o_i,mu_p_e,mu_h_e,mu_o_e)
    
def conductivity(den_t):        #(s_p,-s_h//s_h,s_p)
    f=mobility()
    mu_p_i=f[0];mu_h_i=f[1];mu_o_i=f[2];
    mu_p_e=f[3];mu_h_e=f[4];mu_o_e=f[5];    

    sigma_p=q_charge*den_t*(mu_p_i-mu_p_e)
    sigma_h=q_charge*den_t*(mu_h_i-mu_h_e)
    sigma_o=q_charge*den_t*(mu_o_i-mu_o_e)
    
    print 'CONDUCTIVITY',sigma_p.max(),sigma_h.max(),sigma_o.max()
#    fig = figure(3,figsize=(18,12),facecolor='w',edgecolor='k')
#    semilogx(sigma_p[:,0,0]*1.e+06,alt_i);semilogx(sigma_h[:,0,0]*1.e+06,alt_i);semilogx(sigma_o[:,0,0],alt_i);legend(('$\sigma_p$','$\sigma_h$','$\sigma_o$'))
    return (sigma_p,sigma_h,sigma_o)

    
#=============================================================================#    

def campo_ele_sph_coord(t_e,wp_g,wf_g,wq_g,den_t,pot):
    global up_tot,uf_tot,uq_tot
    
    angle_I=theta[np-np_i:,:,:]
#===
    f=conductivity(den_t)
    sigma_p=f[0];sigma_h=f[1];sigma_o=f[2];
    
    s_11=sigma_p;s_12=-sigma_h*cos(angle_I);s_13=-sigma_h*sin(angle_I)
    s_21=-s_12;s_22=sigma_p*cos(angle_I)**2.+sigma_o*sin(angle_I)**2.;s_23=(sigma_p-sigma_o)*sin(angle_I)*cos(angle_I)
    s_31=-s_12;s_32=s_23;s_33=sigma_o*cos(angle_I)**2.+sigma_p*sin(angle_I)**2.    
    
#===   
    f=mobility()
    mu_p_i=f[0];mu_h_i=f[1];mu_o_i=f[2]
    mu_p_e=f[3];mu_h_e=f[4];mu_o_e=f[5]

    kappa=f[2]*mag_o
    mu_11=f[0];mu_12=-mu_11*kappa*cos(angle_I);mu_13=-mu_11*kappa*sin(angle_I)
    mu_21=-mu_12;mu_22=mu_11*(1.+kappa**2.*sin(angle_I)**2.);mu_23=-mu_11*kappa**2.*cos(angle_I)*sin(angle_I)   
    mu_31=-mu_13;mu_32=mu_23;mu_33=mu_11*(1.+kappa**2.*cos(angle_I)**2.)
    
    ele_w_p=-wf_g*mag_o;ele_w_f=wp_g*mag_o;ele_w_q=wq_g*mag_o
    
    up_agw_i=mu_11*ele_w_p+mu_12*ele_w_f+mu_13*ele_w_q
    uf_agw_i=mu_22*ele_w_f+mu_21*ele_w_p+mu_23*ele_w_q
    uq_agw_i=mu_33*ele_w_q+mu_31*ele_w_p+mu_32*ele_w_f


    kappa=f[5]*mag_o
    mu_11=f[3];mu_12=-mu_11*kappa*cos(angle_I);mu_13=-mu_11*kappa*sin(angle_I)
    mu_21=-mu_12;mu_22=mu_11*(1.+kappa**2.*sin(angle_I)**2.);mu_23=-mu_11*kappa**2.*cos(angle_I)*sin(angle_I)   
    mu_31=-mu_13;mu_32=mu_23;mu_33=mu_11*(1.+kappa**2.*cos(angle_I)**2.)
    
    ele_w_p=-wf_g*mag_o;ele_w_f=wp_g*mag_o;ele_w_q=wq_g*mag_o
    
    up_agw_e=mu_11*ele_w_p+mu_12*ele_w_f+mu_13*ele_w_q
    uf_agw_e=mu_22*ele_w_f+mu_21*ele_w_p+mu_23*ele_w_q
    uq_agw_e=mu_33*ele_w_q+mu_31*ele_w_p+mu_32*ele_w_f
    
    up_ele=0;uf_ele=0;uq_ele=0
#===
    
    up_tot=up_ele+up_agw_e;uf_tot=uf_ele+uf_agw_e;uq_tot=uq_ele+uq_agw_e

    print 'POTENTIAL, U_MAX', round(up_tot.max(),2),round(uf_tot.max(),2),round(uq_tot.max(),2) 
    return (pot,up_ele)


#=============================================================================#
    
def den_ion(den_t):
    
    den_o=den_t
    
    den_g=den_t
    for i_sor in range (0,8):
        flux_p=den_g*up_tot;flux_f=den_g*uf_tot;flux_q=den_g*uq_tot;
        gr_p=gradient(flux_p);gr_f=gradient(flux_f);gr_q=gradient(flux_q)
    
        div_flux=gr_p[0]/dp_i+gr_f[1]/df_i+gr_q[2]/dq_i
    
        den_t=den_o-dt*(div_flux)
        den_g=den_t
        
    delta_den=(den_t-den_amb)/den_t
    dtec=tec_fac*(den_t.sum(0)-den_amb.sum(0))
#===
    
    print 'ION_DENSITY',round(100.*delta_den.max(),2)    
    return (den_t,delta_den,dtec)
    
#=============================================================================#
#=============================================================================#

def AGW_2(i_axis,delta,press,rho,rho_d,rho_u,div_w,div_flux,w_gr_press):
    gamma=1.4
    flux_p=gamma*press*div_w
    grad_temp=gradient(flux_p);grad_flux=grad_temp[i_axis]/delta
    w_1=grad_flux/rho 
    
    grad_temp=gradient(press);grad_press=abs(grad_temp[i_axis])/delta    #IMPORTANT OF USING ABS
    rho_m=(rho_u+2.*rho+rho_d)/4.;
    w_2=-grad_press*div_flux/rho_m**2.;        
    
    grad_temp=gradient(w_gr_press); 
    w_3=grad_temp[i_axis]/delta; 
    
    return (w_1,w_2,w_3)    

#=============================================================================#
#==================================MAIN=======================================#

global t

rho=zeros((np,nf,nq));rho_d=zeros((np,nf,nq));rho_u=zeros((np,nf,nq));
temp=zeros((np,nf,nq));r_g=zeros((np,nf,nq))


f=atmos(alt)
rho_o=f[0];rho_od=f[1];rho_ou=f[2];temp_o=f[3]

for k in range (0,nq):
    for j in range (0,nf):
        rho[:,j,k]=rho_o;rho_d[:,j,k]=rho_od;rho_u[:,j,k]=rho_ou;
        temp[:,j,k]=temp_o;
        r_g[:,j,k]=150.*(1.+sqrt(alt+5.)/5.);


nt=400;t=zeros((nt))
wp_m=zeros((np,nf,nq));wf_m=zeros((np,nf,nq));wq_m=zeros((np,nf,nq));
wp_o=zeros((np,nf,nq));wf_o=zeros((np,nf,nq));wq_o=zeros((np,nf,nq));
wp=zeros((np,nf,nq));wf=zeros((np,nf,nq));wq=zeros((np,nf,nq));
up_tot=zeros((np_i,nf,nq))
i_t=1    
#for i_t in range (1,nt-1):    
while logical_and(i_t < nt-1,logical_and(wp.max()<500,up_tot.max()<1500.)):
    global dt
    w_g=1.
    for i_sor in range (0,3):
        
        #=====================================================================#        
#        dt=2*int(0.5*dp_m.min()/c_s.max());
        dt=20.;
        t[1]=t[0]+dt
        
        #=====================================================================#
        
        gamma=1.4;press=r_g*rho*temp;c_s=sqrt(gamma*press/rho);
        
    
        #=====================================================================#
        
        if i_sor ==0:
            f=source_t(t[i_t]-dt)
            s_t=f
            wp_m=wp_m+1.e-03*s_t*source_shape
    
            f=source_t(t[i_t])
            s_t=f
            wp_o=wp_o+1.e-03*s_t*source_shape
    
            f=source_t(t[i_t]+dt)
            s_t=f
            wp=1.e-03*s_t*source_shape        
    
        if logical_and(i_t==1,i_sor==0):
            wp_g=wp_o;wf_g=wf_o;wq_g=wq_o;    
        
        #=====================================================================#
        
        wp_0=2.*wp_o-wp_m;
        wf_0=2.*wf_o-wf_m;
        wq_0=2.*wq_o-wq_m;
        
        #=====================================================================#
        
        gr_p=gradient(wp_g);gr_f=gradient(wf_g);gr_q=gradient(wq_g)
        div_w=gr_p[0]/dp_m+gr_f[1]/df_m+gr_q[2]/dq_m
        
        flux_p=rho*wp_g;flux_f=rho*wf_g;flux_q=rho*wq_g;
        gr_p=gradient(flux_p);gr_f=gradient(flux_f);gr_q=gradient(flux_q)
        div_flux=gr_p[0]/dp_m+gr_f[1]/df_m+gr_q[2]/dq_m
        
        gr_pr=gradient(press)
        w_gr_press=wp_g*gr_pr[0]/dp_m+wf_g*gr_pr[1]/df_m+wq_g*gr_pr[2]/dq_m
    
        #=====================================================================#

        if i_t==1:rho_0=rho;
           
        f_p=AGW_2(0,dp_m,press,rho,rho_d,rho_u,div_w,div_flux,w_gr_press);
        f_f=AGW_2(1,df_m,press,rho,rho_d,rho_u,div_w,div_flux,w_gr_press);
        f_q=AGW_2(2,dq_m,press,rho,rho_d,rho_u,div_w,div_flux,w_gr_press);        
        wp_1=f_p[0];wp_2=f_p[1];wp_3=f_p[2]
        wf_1=f_f[0];wf_2=f_f[1];wf_3=f_f[2]
        wq_1=f_q[0];wq_2=f_q[1];wq_3=f_q[2]
        
        #=====================================================================#
        
        visc_mu=3.563e-07*temp**(0.71);visc_ki=visc_mu/rho;
        a=1./dp_m**2.
        flux_p=rho*wp_o
        grad_temp=gradient(flux_p);grad_flux=grad_temp[0]/dp_m
        w_visc_p=a*grad_flux*visc_mu/rho**2.      
        a=1./df_m**2.
        flux_p=rho*wf_o
        grad_temp=gradient(flux_p);grad_flux=grad_temp[1]/df_m
        w_visc_f=a*grad_flux*visc_mu/rho**2.
        a=1./dq_m**2.
        flux_p=rho*wq_o
        grad_temp=gradient(flux_p);grad_flux=grad_temp[2]/dq_m
        w_visc_q=a*grad_flux*visc_mu/rho**2.
        
        w_visc_rho=abs(w_visc_p)+abs(w_visc_f)+abs(w_visc_q)

        a=1./dp_m**2.+1./df_m**2.+1./dq_m**2.
        w_visc_w=visc_ki*a/dt

        w_visc=abs(w_visc_rho)+abs(w_visc_w)
        
        w_damp=exp(-0.5*dt**2.*w_visc)
        
        #=====================================================================#
        
        fac_nl=wp_g.max()**2./(dp_m*dt)
        w_nl=exp(-0.1*dt**2.*fac_nl)
        wp_amp=w_nl*w_damp*mask_bound    
        
        fac_nl=wf_g.max()**1./(df_m*dt)+wq_g.max()**1./(dq_m*dt)
        w_nl=exp(-0.001*dt**2.*fac_nl)
        w_amp=w_nl*w_damp*mask_bound  
        
        #=====================================================================#
        
        wp[0:,:,:]=wp_amp[0:,:,:]*(wp_0[0:,:,:]+dt**2.*(wp_1[0:,:,:]+wp_2[0:,:,:]+wp_3[0:,:,:]));
        wf=w_amp*(wf_0+dt**2.*(wf_1+wf_2+wf_3));
        wq=w_amp*(wq_0+dt**2.*(wq_1+wq_2+wq_3));
        
        #=====================================================================#
        
        error=abs(wp-wp_g)/wp.max()
        wp_g=wp;wf_g=wf;wq_g=wq
        
    
        w_g=w_g-0.1
        if w_g < 1.:w_g=1.
             
        print "SOR LOOP CONTINUES", i_sor,w_g,error.max()
        
    #=========================================================================#
    
    wp_m[0:,:,:]=wp_o[0:,:,:];wp_o[0:,:,:]=wp[0:,:,:]
    wf_m=wf_o;wf_o=wf
    wq_m=wq_o;wq_o=wq
    
    #=========================================================================#
    #=========================================================================#
    for i_sor in range (0,1):
        if i_sor==0:
            rho_oo=rho;temp_oo=temp
        flux_p=rho*wp_g;flux_f=rho*wf_g;flux_q=rho*wq_g;
        gr_p=gradient(flux_p);gr_f=gradient(flux_f);gr_q=gradient(flux_q)
        div_flux=gr_p[0]/dp_m+gr_f[1]/df_m+gr_q[2]/dq_m
        
        rho=rho_oo-0.*dt*(div_flux)
        if any(rho<0):
            rho=rho_oo
        rho_d[0:np-1,:,:]=rho[1:np,:,:];rho_u[1:np,:,:]=rho[0:np-1,:,:]
            
        flux_p=temp*wp_g;flux_f=temp*wf_g;flux_q=temp*wq_g;
        gr_p=gradient(flux_p);gr_f=gradient(flux_f);gr_q=gradient(flux_q)
        div_flux=gr_p[0]/dp_m+gr_f[1]/df_m+gr_q[2]/dq_m
        
        temp=temp_oo-0.25*dt*(div_flux)

    #=========================================================================#
    #===========================IONOSPHERE====================================#

    #========================ION_CONTINUITY===================================#
    f=iono_amb()    
    if i_t==1:
        den_t=den_amb;
        pot=zeros((np_i,nf,nq));
    wp_i=wp_g[np-np_i:,:,:];wf_i=wf_g[np-np_i:,:,:];wq_i=wq_g[np-np_i:,:,:]
    
    if i_mag==0:
        f=campo_ele_sph_coord(t[i_t],wp_i,wf_i,wq_i,den_t,pot)    
        pot=f[0];up_ele=f[1]
    
    if i_mag==1:
        f=campo_ele_mag_coord(t[i_t],wp_i,wf_i,wq_i,den_t,pot)    
        pot=f[0];up_ele=f[1]
        
    f=den_ion(den_t)
    den_t=f[0];delta_den=f[1];dtec=f[2]
    
    f=conductivity(den_amb)
    sigma_amb=f[2]
#    f=conductivity(den_t)
#    sigma_o=f[2]-sigma_amb
    #=========================================================================#
    #=========================================================================#

    wh=sqrt(wf**2.+wq**2.);
    
    if i_t==1:
        dtec_keo=zeros((int(nt/5),nf,nq));uplift_keo=zeros((int(nt/5),nf,nq));
        wp_keo=zeros((int(nt/5),nf,nq));wh_keo=zeros((int(nt/5),nf,nq));
        pot_keo=zeros((int(nt/5),nf,nq));                
        i_w=0
    if remainder(i_t,5)==0:
        dtec_keo[i_w,:,:]=dtec;uplift_keo[i_w,:,:]=wp[0,:,:];
        wp_keo[i_w,:,:]=wp[13,:,:];wh_keo[i_w,:,:]=wh[13,:,:];
        pot_keo[i_w,:,:]=pot[3,:,:]
        i_w=i_w+1
       
    if i_t==1:
        wp_prop=zeros((nt,np));uplift_t=zeros((nt,np))
    wp_prop[i_t,:]=wp[:,iphi_s,iq_s];uplift_t[i_t,:]=wp[0,iphi_s,iq_s];
    
#    if remainder(i_t,50)==0:
#        fname='wp_3_%05d.npy' % int(t[i_t])
#        save(fname,wp)
#        fname='up_3_%05d.npy' % int(t[i_t])
#        save(fname,up_ele)
#        fname='den_3_%05d.npy' % int(t[i_t])
#        save(fname,den_t)        
    #=========================================================================#
    
    vm=0.05
    fr_gr_2=wp[0,:,:]
    if remainder(i_t,5)==0:
        fig = figure(2,figsize=(18,12),facecolor='w',edgecolor='k') 
        lat_pl=lat_3[5+11,:,:];lon_pl=lon_3[5+11,:,:];
        ax=subplot(121)
#        map=Basemap(llcrnrlon=lon.min(),llcrnrlat=lat_geo.min(),urcrnrlon=lon.max(),urcrnrlat=lat_geo.max(),suppress_ticks=False)
#        map.etopo(alpha=0.5)
        map = Basemap(projection='ortho',lat_0=lat_pl.mean(),lon_0=lon_pl.mean(),resolution='l')
        map.etopo(alpha=0.5)
        map.drawmeridians(arange(lon_pl.min(),lon_pl.max()+10,10))
        map.drawparallels(arange(lat_pl.min(),lat_pl.max()+10,10))
        x, y = map(lon_pl,lat_pl)
        contour(x,y,abs(fr_gr_2)/fr_gr_2.max(),colors='g',linewidths=1,levels=[0.05,0.5],alpha=0.2)
        data=wp[5+11,:,:]
        im=pcolormesh(x,y,data/data.max(),cmap=cm.seismic,vmax=vm,vmin=-vm,alpha=0.5);
        xlabel('Longitude, $^o$');ylabel('Latitude, $^o$')        
        title(str.join(' ',['AGW: MAX=',str(data.max())]))
        axis('off')
        divider=make_axes_locatable(ax);cax=divider.append_axes("right", size="2%", pad=0.05)
        colorbar(im,cax=cax); 

        
        ax=subplot(122)
#        map=Basemap(llcrnrlon=lon.min(),llcrnrlat=lat_geo.min(),urcrnrlon=lon.max(),urcrnrlat=lat_geo.max(),suppress_ticks=False)
#        map.etopo(alpha=0.5)
        map = Basemap(projection='ortho',lat_0=lat_pl.mean(),lon_0=lon_pl.mean(),resolution='l')
        map.etopo(alpha=0.5)
        map.drawmeridians(arange(lon_pl.min(),lon_pl.max()+10,10))
        map.drawparallels(arange(lat_pl.min(),lat_pl.max()+10,10))
        x, y = map(lon_pl,lat_pl)
        contour(x,y,abs(fr_gr_2)/fr_gr_2.max(),colors='g',linewidths=1,levels=[0.05,0.5],alpha=0.2)
        data=pot[11,:,:]
        con_lev=[-0.5,-0.05,-0.00005,0.00005,0.05,0.5]
        cs=contour(zoom(x,2),zoom(y,2),zoom(data/data.max(),2),colors='k',linewidths=1,levels=con_lev); 
        clabel(cs,fontsize=8)

        data=dtec
        im=pcolormesh(x,y,data/data.max(),cmap=cm.seismic,vmax=vm,vmin=-vm,alpha=0.5);
        xlabel('Longitude, $^o$');ylabel('Latitude, $^o$')        
        title(str.join(' ',['$\Delta$ TEC: MAX=',str(data.max())]))
        axis('off')
        divider=make_axes_locatable(ax);cax=divider.append_axes("right", size="2%", pad=0.05)
        colorbar(im,cax=cax); 

        draw();
#
#        fname='figura_PI_I_%05d.png' % int(t[i_t])
#        savefig(fname)
        pause(0.1);clf()
#===
#===                
        
        fig = figure(3,figsize=(18,12),facecolor='w',edgecolor='k') 
        
        subplot(121)
        data=wp[:,iphi_s,:];lat_geo=z_sph[:,iphi_s,:];alt_geo=y_sph[:,iphi_s,:]     
        im=pcolormesh(zoom(lat_geo,2),zoom(alt_geo,2),zoom(data/data.max(),2),cmap=cm.seismic,vmax=vm,vmin=-vm);axis('tight')
        con_lev=[-0.5,-0.1,-0.05,0.05,0.1,0.5]
        contour(zoom(lat_geo,2),zoom(alt_geo,2),zoom(data/data.max(),2),colors='g',levels=con_lev);axis('tight')
        
        plot(z_f[:,0,0:nq:10],y_f[:,0,0:nq:10],color='gray',alpha=0.2)
        labels=around(lat[0:nq:10],decimals=0)
        ii=0
        for i in range (0,nq,10):
            text(z_f[np-1,0,i],y_f[np-1,0,i],str(int(labels[ii])),fontsize=8)
            ii=ii+1
        plot(z_f[0:np:10,0,:].T,y_f[0:np:10,0,:].T,color='gray',alpha=0.2)
        labels=around(alt[0:np:10],decimals=0)
        ii=0
        for i in range (0,np,10):
            text(z_f[i,0,int(nq/2.)],y_f[i,0,int(nq/2.)],str(int(labels[ii])),fontsize=8)
            ii=ii+1

        gca().add_patch(Circle((0,0),radius=r_ea,fc='gray',fill=False,alpha=1))
        gca().add_patch(Circle((0,0),radius=r_ea-100,fc='w',fill=False,alpha=1))
        axis('off')        
        title(str.join(' ',['AGW: MAX=',str(data.max())]))

#====
        
        ax=subplot(122)
        data=pot[:,iphi_s,:];lat_geo=z_sph[np-np_i:,iphi_s,:];alt_geo=y_sph[np-np_i:,iphi_s,:]      
        con_lev=[-0.5,-0.05,-0.00005,.00005,0.05,0.5]
        contour(zoom(lat_geo,2),zoom(alt_geo,2),zoom(data/data.max(),2),colors='g',levels=con_lev);axis('tight')        
        title(str.join(' ',['$POT$: MAX=',str(data.max())]))        
        
        data=den_t[:,iphi_s,:];#100*delta_den[:,iphi_s,:];
        im=pcolormesh(zoom(lat_geo,2),zoom(alt_geo,2),zoom(data/data.max(),2),cmap=cm.gray)#,vmax=vm,vmin=-vm);axis('tight')
        
        plot(z_f[:,0,0:nq:10],y_f[:,0,0:nq:10],color='gray',alpha=0.2)
        labels=around(lat[0:nq:10],decimals=0)
        ii=0
        for i in range (0,nq,10):
            text(z_f[np-1,0,i],y_f[np-1,0,i],str(int(labels[ii])),fontsize=8)
            ii=ii+1
        plot(z_f[0:np:10,0,:].T,y_f[0:np:10,0,:].T,color='gray',alpha=0.2)
        labels=around(alt[0:np:10],decimals=0)
        ii=0
        for i in range (0,np,10):
            text(z_f[i,0,int(nq/2.)],y_f[i,0,int(nq/2.)],str(int(labels[ii])),fontsize=8)
            ii=ii+1
            
        gca().add_patch(Circle((0,0),radius=r_ea,fc='gray',fill=False,alpha=1))
        gca().add_patch(Circle((0,0),radius=r_ea-100,fc='w',fill=False,alpha=1)) 
        axis('off')        
    
        divider=make_axes_locatable(ax);cax=divider.append_axes("right", size="2%", pad=0.05)
        colorbar(im,cax=cax); 
        
        draw();
#        fname='figura_PI_II_%05d.png' % int(t[i_t])
#        savefig(fname)
        pause(0.1);clf()
        
        fig = figure(4,figsize=(8,12),facecolor='w',edgecolor='k') 
        data=up_tot[:,:,iq_s]        
        pcolormesh(zoom(lon,2),zoom(alt_i,2),zoom(data,2),cmap=cm.seismic)
        data=den_t[:,:,iq_s];
        contour(zoom(lon,2),zoom(alt_i,2),zoom(data,2),21,colors='g');axis('tight')
        
        draw();pause(0.1);clf()
        
    print "PLOTTING"
    
    
    print '======================================================'
    print "TIME=", i_t,t[i_t]
    print 'FORCING AMPLITUDE=',round(wp[0,:,:].max(),5)
    print 'ATMOS W_MAX=',round(wp.max(),2),round(wf.max(),2),round(wq.max(),2) 
    
    t[i_t+1]=t[i_t]+dt;
    i_t=i_t+1
    
#=============================================================================#
#=============================================================================#
    
#save('uplift_t.npy',uplift_t)
#save('wp_prop_t.npy',wp_prop)
#
save('uplift_keo_t.npy',uplift_keo)
save('wp_keo_t.npy',wp_keo)
save('wh_keo_t.npy',wh_keo)
save('dtec_keo_t.npy',dtec_keo)


#=============================================================================#
#=============================================================================#

#tsu_t=load('uplift_t.npy')
#wp_t=load('wp_prop_t.npy');nt=len(wp_t[:,0])
#time=arange(0,20*nt,20);alt=arange(0,480,15);
#fig = figure(3,figsize=(18,12),facecolor='w',edgecolor='k') 
#plot(time,tsu_t/tsu_t.max(),'g')
##plot(time,wp_t[:,3])
##plot(time,wp_t[:,6])
##plot(time,wp_t[:,9])
#plot(time,wp_t[:,12],'r')
#pcolormesh(time,alt,wp_t.T,cmap=cm.seismic);axis('tight');

#=============================================================================#
#=============================================================================#


show()