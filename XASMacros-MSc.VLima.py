import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import integrate
from matplotlib import ticker
%matplotlib notebook
plt.rcParams['figure.dpi']=150
plt.rcParams['figure.figsize']=(6,4)

def XASload(dir,de,ate,ext,interp,groupA,groupB,exc,y,shift):
          
    a=[];b=[];
  
    for i in range(de,ate+1):
            
        c = '{:04d}'.format(i)
        
        if '2' in ext:
            energy_i,col2,col3,col4,i0_i,sample_i,norm_i,col8,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=1,delimiter=',',unpack=True)
            
            if i == de: 
                energy = np.arange(round(energy_i[0]+2,3),round(energy_i[-1]-2,3),0.01)   
           
            interp_norm_i = interp1d(energy_i, norm_i, kind='linear')
            norm = interp_norm_i(energy)
            
            interp_i0_i = interp1d(energy_i, i0_i, kind='linear')
            i0 = interp_i0_i(energy)
            
            interp_sample_i = interp1d(energy_i, sample_i, kind='linear')
            sample = interp_sample_i(energy)  
            
        elif interp == 0:
            col0,energy_i,col2,col3,col4,col5,i0_i,col7,sample_i,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=7,delimiter=',',unpack=True)
            energy = energy_i
            norm = np.array(sample_i/i0_i)
            i0 = i0_i
            sample = sample_i
        else:
            col0,energy_i,col2,col3,col4,col5,i0_i,col7,sample_i,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=7,delimiter=',',unpack=True)
            norm_i = np.array(sample_i/i0_i)

            if i == de: 
                energy = np.arange(round(energy_i[0]+2,3),round(energy_i[-1]-2,3),0.01)   
            # JCC em 20180412: reduzimos o intervalo de energia em 1% no começo e no final para evitar erros de interpolação... Tem que melhorar...    
            
            interp_norm_i = interp1d(energy_i, norm_i, kind='linear')
            norm = interp_norm_i(energy)
            
            interp_i0_i = interp1d(energy_i, i0_i, kind='linear')
            i0 = interp_i0_i(energy)
            
            interp_sample_i = interp1d(energy_i, sample_i, kind='linear')
            sample = interp_sample_i(energy)            
           
        if y == 0:    yaxis = i0;   xaxis = energy
        elif y == 1:  yaxis = sample;   xaxis = energy
        elif y == 2:    yaxis = norm;   xaxis = energy;  
        
        elif y == 11:
            energy_shift,energy_cut,sample_cut,popt,amp,x0,sigma,gauss = XASshift(energy,sample,ref=shift[0],left=shift[1],right=shift[2])
            if i == de: 
                energy_new = np.arange(round(energy_shift[0],3),round(energy_shift[-1],3),0.05)
            interp_sample = interp1d(energy_shift, sample, kind='linear', fill_value='extrapolate')
            sample_new = interp_sample(energy_new)
            yaxis = sample_new; xaxis = energy_new
            
        elif y == 21:
            energy_shift,energy_cut,norm_cut,popt,amp,x0,sigma,gauss = XASshift(energy,norm,ref=shift[0],left=shift[1],right=shift[2])
            if i == de: 
                energy_new = np.arange(round(energy_shift[0],3),round(energy_shift[-1],3),0.05)
            interp_norm = interp1d(energy_shift, norm, kind='linear', fill_value='extrapolate')
            norm_new = interp_norm(energy_new)
            yaxis = norm_new; xaxis = energy_new                                          
                         
        c = int(c)
        t = c - de + 1        
            
        if (t in groupA) and (t not in exc):    
            a.append(yaxis)
        elif (t in groupB) and (t not in exc):    
            b.append(yaxis)
        
    return a,b,xaxis

def SEARCHindex(axis,pt):
    
    test = []
    
    for i in axis:
        b = abs(pt - i)
        test.append(round(b,1))
    idx = test.index(min(test))
    
    return idx


def XASavg(a):    
    
    sum = 0
    
    for i in a:
        sum += i       
            
    avg = sum/(len(a))             
    
    return avg


def XASbg(yaxis,xaxis,bg): 
    
    idx = SEARCHindex(xaxis,bg)
        
    bg = yaxis[idx]
    
    yaxis_bg = yaxis - bg
    
    return yaxis_bg


def XASnor(yaxis,xaxis,xas,nor):   
                                    
    idx = SEARCHindex(xaxis,nor)
 
    yaxis_nor = yaxis/(xas[idx])
    
    return yaxis_nor

def CONFIGplt(ylabel='norm',xlabel='energy (eV)',grid=True):
    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(False)
    plt.legend() 
    plt.show()
    
def XASshift(xaxis,yaxis,ref=779.72,left=-0.5,right=0.5): #ex: energy_shift,energy_cut,norm_cut,popt,a,x0,sigma,gauss = XASshift(energy,norm,ref=779.72,left=0.5,right=0.5)
    
    #defining the gaussian function
    
    def gauss(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    #cutting the data around L3
    
    ref_ini = ref + left
    ref_fin = ref + right
    
    test_ini = []
    test_fin = []
    for i in xaxis:
        a = abs(ref_ini - i)
        test_ini.append(round(a,1))
        b = abs(ref_fin - i)
        test_fin.append(round(b,1))
    ref_ini_idx = test_ini.index(min(test_ini))
    ref_fin_idx = test_fin.index(min(test_fin))

    yaxis_cut = yaxis[int(ref_ini_idx):int(ref_fin_idx)]
    xaxis_cut = xaxis[int(ref_ini_idx):int(ref_fin_idx)]

    #fitting the peak

    popt,pcov = curve_fit(gauss,xaxis_cut,yaxis_cut,p0=[max(yaxis),ref,1],bounds=([min(yaxis),ref_ini,0],[max(yaxis),ref_fin,5]))
    a,x0,sigma = popt[0],popt[1],popt[2]
    
    #shifting the xaxis
    
    shift = ref - x0
    xaxis_shift = xaxis + shift 
    
    return xaxis_shift,xaxis_cut,yaxis_cut,popt,a,x0,sigma,gauss(xaxis,a,x0,sigma)


def XASshiftEXPORT(dir,ext='.dat',scans=[],nor=779.7,bg=772,y=21,shift=[779.7,-0.5,0.5]): #ex: energy_shift,energy_cut,norm_cut,popt,a,x0,sigma,gauss = XASshift(energy,norm,ref=779.72,left=0.5,right=0.5)
    
    for i in scans:
            
        c = '{:04d}'.format(i)
        
        if '2' in ext:
            energy_i,col2,col3,col4,i0_i,sample_i,norm_i,col8,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=1,delimiter=',',unpack=True)
            
            if i == scans[0]: 
                energy = np.arange(round(energy_i[0],3),round(energy_i[-1],3),0.05)             
            
            interp_norm_i = interp1d(energy_i, norm_i, kind='linear', fill_value='extrapolate')
            norm = interp_norm_i(energy)
            
            #interp_i0_i = interp1d(energy_i, i0_i, kind='linear', fill_value='extrapolate')
            #i0 = interp_i0_i(energy)
            
            interp_sample_i = interp1d(energy_i, sample_i, kind='linear', fill_value='extrapolate')
            sample = interp_sample_i(energy)  
            
        else:
            col0,energy_i,col2,col3,col4,col5,i0_i,col7,sample_i,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=7,delimiter=',',unpack=True)
            energy = energy_i
            norm = np.array(sample_i/i0_i)
            #i0 = i0_i
            sample = sample_i           
        
        if y == 11:
            energy_shift,energy_cut,sample_cut,popt,amp,x0,sigma,gauss = XASshift(energy,sample,ref=shift[0],left=shift[1],right=shift[2])
        elif y == 21:
            energy_shift,energy_cut,sample_cut,popt,amp,x0,sigma,gauss = XASshift(energy,norm,ref=shift[0],left=shift[1],right=shift[2])
      
        if i == scans[0]: 
            energy_new = np.arange(round(energy_shift[0],3),round(energy_shift[-1],3),0.05)
            xaxis = energy_new
            
        if y == 11:        
            interp_sample = interp1d(energy_shift, sample, kind='linear', fill_value='extrapolate')
            sample_new = interp_sample(energy_new)
            yaxis = sample_new            
        elif y == 21:          
            interp_norm = interp1d(energy_shift, norm, kind='linear', fill_value='extrapolate')
            norm_new = interp_norm(energy_new)
            yaxis = norm_new
    
        filename = dir+c+'-SHIFTexport.dat'
        head = '#xaxis_new,yaxis_new\n'   
        
        file = open(filename,'w')
        
        file.write(head)
        
        for i in range(len(xaxis)):
            line = str(xaxis[i])+','+str(yaxis[i])+'\n'
            file.write(line)
            
        file.close()
    
def XASplot(dir,scans=[],ext='.dat',marker='',y=2,shift=[779.72,-0.5,0.5],ymin=888,ymax=888):   
    
    for i in scans:
        
        c = '{:04d}'.format(i)
            
        if '2' in ext:
            energy,col2,col3,col4,i0,sample,norm,col8,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=1,delimiter=',',unpack=True)
        else:
            col0,energy,col2,col3,col4,col5,i0,col7,sample,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=7,delimiter=',',unpack=True)
            norm = np.array(sample/i0)
        
        if y == 0:    yaxis = i0;   ylab = 'i0';    xaxis = energy
        elif y == 1:  yaxis = sample;   ylab = 'sample';    xaxis = energy
        elif y == 2:    yaxis = norm;   ylab = 'norm';    xaxis = energy
        elif y == 11:
            energy_shift,energy_cut,sample_cut,popt,amp,x0,sigma,gauss = XASshift(energy,sample,ref=shift[0],left=shift[1],right=shift[2])
            yaxis = sample;   ylab = 'sample';    xaxis = energy_shift 
        elif y == 21:
            energy_shift,energy_cut,norm_cut,popt,amp,x0,sigma,gauss = XASshift(energy,norm,ref=shift[0],left=shift[1],right=shift[2])
            yaxis = norm;   ylab = 'norm';    xaxis = energy_shift 
            
        plt.plot(xaxis, yaxis, linestyle='-',linewidth=1.2, label=str(i), marker=marker)
        plt.grid(False)

        
    if ymin!= 888:
        plt.ylim((ymin,ymax))    
        CONFIGplt(ylabel=ylab)              
    else:
        CONFIGplt(ylabel=ylab) 
        
def XASplotAVG(dir,scans=[],ext='.dat',marker='',y=2,shift=[779.72,-0.5,0.5],ymin=888,ymax=888): 
             
    a=[]
    
    for i in scans:
        
        c = '{:04d}'.format(i)
            
        if '2' in ext:
            energy,col2,col3,col4,i0,sample,norm,col8,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=1,delimiter=',',unpack=True)
        else:
            col0,energy,col2,col3,col4,col5,i0,col7,sample,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=7,delimiter=',',unpack=True)
            norm = np.array(sample/i0)
        
        if y == 0:    yaxis = i0;   ylab = 'i0';    xaxis = energy
        elif y == 1:  yaxis = sample;   ylab = 'sample';    xaxis = energy
        elif y == 2:    yaxis = norm;   ylab = 'norm';    xaxis = energy
        elif y == 11:
            energy_shift,energy_cut,sample_cut,popt,amp,x0,sigma,gauss = XASshift(energy,sample,ref=shift[0],left=shift[1],right=shift[2])
            yaxis = sample;   ylab = 'sample';    xaxis = energy_shift 
        elif y == 21:
            energy_shift,energy_cut,norm_cut,popt,amp,x0,sigma,gauss = XASshift(energy,norm,ref=shift[0],left=shift[1],right=shift[2])
            yaxis = norm;   ylab = 'norm';    xaxis = energy_shift 
            
        a.append(yaxis)
    
    media = XASavg(a)
    plt.plot(xaxis, yaxis, linestyle='-',linewidth=1.2, label=str(i), marker=marker)
        
    if ymin!= 888:
        plt.ylim((ymin,ymax))    
        CONFIGplt(ylabel=ylab)              
    else:
        CONFIGplt(ylabel=ylab)
        
def XASplotBGnor(dir,scans=[],ext='.dat',marker='',y=2,shift=[779.72,-0.5,0.5],ymin=888,ymax=888,bg=775,nor=777.9): 
             
    for i in scans:
        
        c = '{:04d}'.format(i)
            
        if '2' in ext:
            energy,col2,col3,col4,i0,sample,norm,col8,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=1,delimiter=',',unpack=True)
        else:
            col0,energy,col2,col3,col4,col5,i0,col7,sample,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=7,delimiter=',',unpack=True)
            norm = np.array(sample/i0)
        
        if y == 0:    yaxis = i0;   ylab = 'i0';    xaxis = energy
        elif y == 1:  yaxis = sample;   ylab = 'sample';    xaxis = energy
        elif y == 2:    yaxis = norm;   ylab = 'Normalized Intensity (a. u.)';    xaxis = energy; xlab = 'Energy (eV)'
        elif y == 11:
            energy_shift,energy_cut,sample_cut,popt,amp,x0,sigma,gauss = XASshift(energy,sample,ref=shift[0],left=shift[1],right=shift[2])
            yaxis = sample;   ylab = 'sample';    xaxis = energy_shift 
        elif y == 21:
            energy_shift,energy_cut,norm_cut,popt,amp,x0,sigma,gauss = XASshift(energy,norm,ref=shift[0],left=shift[1],right=shift[2])
            yaxis = norm;   ylab = 'norm';    xaxis = energy_shift 
        
        yaxis_bg = XASbg(yaxis,xaxis,bg)  
        yaxis_bg_nor = XASnor(yaxis_bg,xaxis,yaxis_bg,nor)  
        plt.plot(xaxis, yaxis_bg_nor, linestyle='-',linewidth=1.2,label=str(i),marker=marker)
                
    if ymin!= 888:
        plt.ylim((ymin,ymax))    
        CONFIGplt(ylabel=ylab)              
    else:
        CONFIGplt(ylabel=ylab) 

def XASplotBGnor_export(dir,scans=[],ext='.dat',marker='',y=2,shift=[779.72,-0.5,0.5],ymin=888,ymax=888,bg=775,nor=777.9,name=[]): 

    # Para espectros adquiridos sem polarização/quartetos ou sem dicroísmo. Ex: borda K do oxigênio
    
    for i in scans:
        
        c = '{:04d}'.format(i)
            
        if '2' in ext:
            energy,col2,col3,col4,i0,sample,norm,col8,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=1,delimiter=',',unpack=True)
        else:
            col0,energy,col2,col3,col4,col5,i0,col7,sample,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=7,delimiter=',',unpack=True)
            norm = np.array(sample/i0)
        
        if y == 0:    yaxis = i0;   ylab = 'i0';    xaxis = energy
        elif y == 1:  yaxis = sample;   ylab = 'sample';    xaxis = energy
        elif y == 2:    yaxis = norm;   ylab = 'Normalized Intensity (a. u.)';    xaxis = energy; xlab = 'Energy (eV)'
        elif y == 11:
            energy_shift,energy_cut,sample_cut,popt,amp,x0,sigma,gauss = XASshift(energy,sample,ref=shift[0],left=shift[1],right=shift[2])
            yaxis = sample;   ylab = 'sample';    xaxis = energy_shift 
        elif y == 21:
            energy_shift,energy_cut,norm_cut,popt,amp,x0,sigma,gauss = XASshift(energy,norm,ref=shift[0],left=shift[1],right=shift[2])
            yaxis = norm;   ylab = 'norm';    xaxis = energy_shift 
        
        yaxis_bg = XASbg(yaxis,xaxis,bg)  
        yaxis_bg_nor = XASnor(yaxis_bg,xaxis,yaxis_bg,nor)  
        plt.plot(xaxis, yaxis_bg_nor, linestyle='-',linewidth=1.2,label=str(i),marker=marker)
        plt.axhline(y=0,color='k',linestyle='--',linewidth=0.5)
        plt.grid(False)

                
    if ymin!= 888:
        plt.ylim((ymin,ymax))    
        CONFIGplt(ylabel=ylab)              
    else:
        CONFIGplt(ylabel=ylab) 
        
### vitoraolima july 24 2021
    filename = name+'_XASexport.dat'
    head = '#energy,xas,xas_nor\n'   
    
    file = open(filename,'w')
    
    file.write(head)
    yaxis_bg_nor = XASnor(yaxis_bg,xaxis,yaxis_bg,nor)  
    for i in range(len(energy)):
        line = str(energy[i])+','+str(yaxis_bg[i])+','+str(yaxis_bg_nor[i])+'\n'
        file.write(line)
        
    file.close() 

# This function is the same funcion as "XMCDplot", reported in the end of this macro 
def XAS_and_XMCD_plot(dir,de,ate,ext='.dat',interp=0,pos=[1,4,5,8],neg=[2,3,6,7],exc=[],nor=779.7,bg=772,y=2,shift=[779.7,-0.5,0.5]): 
             
    p,n,energy = XASload(dir,de,ate,ext,interp,pos,neg,exc,y,shift)     
           
    pmedia = XASavg(p) 
    nmedia = XASavg(n)
                
    pmedia_bg = XASbg(pmedia,energy,bg)
    nmedia_bg = XASbg(nmedia,energy,bg)
    
    xas = (pmedia_bg + nmedia_bg)/2
    dif = pmedia_bg - nmedia_bg 
    
    pmedia_bg_nor = XASnor(pmedia_bg,energy,xas,nor)
    nmedia_bg_nor = XASnor(nmedia_bg,energy,xas,nor)    
    xas_nor = XASnor(xas,energy,xas,nor)
    dif_nor = XASnor(dif,energy,xas,nor)
   
    plt.plot(energy, pmedia_bg_nor, linestyle='-', color='black',linewidth=1.5, label='pos_avg'); 
    plt.plot(energy, nmedia_bg_nor, linestyle='-', color='red',linewidth=1.5, label='neg_avg'); 
    plt.plot(energy, xas_nor, linestyle='-', color='green',linewidth=0.5, label='xas_avg'); 
    plt.plot(energy, dif_nor, linestyle='-', color='blue',linewidth=1.5, label='xmcd_asymm'); 
    
    plt.axhline(y=0,color='k',linestyle='--',linewidth=0.5)
    plt.grid(False)
     
    CONFIGplt(ylabel='xas, xmcd asymmetry')
    
# vitoraolima july 24 2021
def XAS_and_XMCD_SaturationEffectsCorrection(dir,de,ate,ext='.dat',interp=0,pos=[1,4,5,8],neg=[2,3,6,7],exc=[],nor=779.7,bg=772,y=2,shift=[779.7,-0.5,0.5],displacement=[],L3=[],l_e=[],thickness=[]): 
             
    p,n,energy = XASload(dir,de,ate,ext,interp,pos,neg,exc,y,shift)     
           
    pmedia = XASavg(p) 
    nmedia = XASavg(n)
                
    pmedia_bg = XASbg(pmedia,energy,bg)
    nmedia_bg = XASbg(nmedia,energy,bg)
    
    xas = (pmedia_bg + nmedia_bg)/2
    dif = pmedia_bg - nmedia_bg 
    
    pmedia_bg_nor = XASnor(pmedia_bg,energy,xas,nor)
    nmedia_bg_nor = XASnor(nmedia_bg,energy,xas,nor)    
    xas_nor = XASnor(xas,energy,xas,nor)
    dif_nor = XASnor(dif,energy,xas,nor)
    
    ### Applying the saturation effects correction in TEY detection
    
    ###### Attenuation length and absorption coeficient extracted from HENKE: importar antes da macro como "Henke.dat"
    Henke = np.loadtxt('raw/Henke.dat',delimiter=',')
    atl_energia = Henke[:,0]+displacement
    atl_intensidade = Henke[:,1]*10**(-6)
    
    mu_energia = atl_energia
    mu_intensidade = 1/atl_intensidade
        
    ###### interpolating in the range of data: 
    number_of_points = len(energy)
    interpolate = sp.interpolate.interp1d(mu_energia,mu_intensidade,fill_value="extrapolate")
    MU_energia = np.linspace(630,670,number_of_points) # range for Mn L2,3 edge: 630,670 Obs: mesmo range dos dados!!!
    MU_intensidade = interpolate(MU_energia)
    
    ###### Background Fitting:
    LinearFit1 = np.polyfit(energy[-70:-1],pmedia_bg[-70:-1],1) 
    H1 = LinearFit1[1]+LinearFit1[0]*energy;
    C = 0
    D = 1; 
    BG1 = C*energy + H1*(1-(1/(1+np.exp((energy-L3)/D))))
    
    LinearFit2 = np.polyfit(energy[-70:-1],nmedia_bg[-70:-1],1) 
    H2 = LinearFit2[1]+LinearFit2[0]*energy;
    C = 0
    D = 1; 
    BG2 = C*energy + H2*(1-(1/(1+np.exp((energy-L3)/D))))
        
    ###### multiplicative factor
    factor1 = MU_intensidade/(BG1+1)
    factor2 = MU_intensidade/(BG2+1)
    
    ###### Absorption coeficient for the data
    ca1 = (pmedia_bg+1)*factor1
    ca2 = (nmedia_bg+1)*factor2
    
    ###### Applying the correction for thin films:  
    
    T = thickness
    
    l_x1 = 1/((ca1))
    F = 1/(1+l_e/(l_x1))
    pmedia_corr = F * (1 - np.exp(-T*((1/l_e)+(1/l_x1))) ) * pmedia_bg
    
    l_x2 = 1/((ca2))
    F = 1/(1+l_e/(l_x2))
    nmedia_corr = F * (1 - np.exp(-T*((1/l_e)+(1/l_x2))) ) * nmedia_bg
    
    ###### Defining the corrected variables
    
    #pmedia = XASavg(p) 
    #nmedia = XASavg(n)
                
    pmedia_bg_corr = XASbg(pmedia_corr,energy,bg)
    nmedia_bg_corr = XASbg(nmedia_corr,energy,bg)
    
    xas_corr = (pmedia_bg_corr + nmedia_bg_corr)/2
    dif_corr = pmedia_bg_corr - nmedia_bg_corr 
    
    pmedia_bg_nor_corr = XASnor(pmedia_bg_corr,energy,xas_corr,nor)
    nmedia_bg_nor_corr = XASnor(nmedia_bg_corr,energy,xas_corr,nor)    
    xas_nor_corr = XASnor(xas_corr,energy,xas_corr,nor)
    dif_nor_corr = XASnor(dif_corr,energy,xas_corr,nor)
    
    ##### plot
    plt.rcParams['figure.dpi']=120
    plt.rcParams['figure.figsize']=(8,4)
    
    plt.figure
    plt.subplot(1,2,1)
    plt.title('Attenuation Length',fontsize=10,weight='bold')
    plt.plot(atl_energia,atl_intensidade,label='From Henke')
    plt.xlabel('Photon energy [eV]')
    plt.ylabel('Attenuation length [m]')
    plt.tight_layout()
    plt.legend(fontsize=10)
    
    plt.subplot(1,2,2)
    plt.title('Absorption coeficient',fontsize=10,weight='bold')
    plt.plot(mu_energia,mu_intensidade,label='From Henke')
    plt.plot(MU_energia,MU_intensidade,'k')
    plt.xlabel('Photon energy [eV]')
    plt.ylabel('Absorption coeficient, $\mu$ [m$^{-1}$]')
    plt.tight_layout()
    plt.legend(fontsize=10)
    
    plt.savefig('dt3.png')
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Background fitting',fontsize=10,weight='bold')
    plt.plot(energy,pmedia_bg,'.',label='pmedia_bg')
    plt.plot(energy,BG1,label='Background')
    plt.plot(energy[-70:-1],H1[-70:-1],label='Linear fit')
    plt.xlabel('Photon energy [eV]')
    plt.ylabel('Intensity [a. u]')
    plt.tight_layout()
    plt.legend(fontsize=10)
        
    plt.subplot(1,2,2)
    plt.title('Background fitting',fontsize=10,weight='bold')
    plt.plot(energy,nmedia_bg,'.',label='nmedia_bg')
    plt.plot(energy,BG2,label='Background')
    plt.plot(energy[-70:-1],H2[-70:-1],label='Linear fit')
    plt.xlabel('Photon energy [eV]')
    plt.ylabel('Intensity [a. u]')
    plt.tight_layout()
    plt.legend(fontsize=10)
    
    plt.savefig('dt4.png')
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Determining the scale factor',fontsize=10,weight='bold')
    plt.plot(MU_energia,MU_intensidade,label='Henke')
    plt.plot(energy,factor1,label='Scale factor for pmedia_bg = Henke/Background')
    plt.plot(energy,(BG1+1)*10**6,label='Background*10$^6$')
    plt.plot(energy,factor1*(BG1+1),'--',label='Proof: scale factor*Background')
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Absorption coeficient, $\mu$ (m$^{-1}$)')
    plt.tight_layout()
    plt.legend(fontsize=6)
    
    plt.subplot(1,2,2)
    plt.title('Determining the scale factor',fontsize=10,weight='bold')
    plt.plot(MU_energia,MU_intensidade,label='Henke')
    plt.plot(energy,factor2,label='Scale factor for nmedia_bg = Henke/Background')
    plt.plot(energy,(BG2+1)*10**6,label='Background*10$^6$')
    plt.plot(energy,factor2*(BG2+1),'--',label='Proof: scale factor*Background')
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Absorption coeficient, $\mu$ (m$^{-1}$)')
    plt.tight_layout()
    plt.legend(fontsize=6)
        
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Adjusted absorption coeficient',fontsize=10,weight='bold')
    plt.plot(energy,ca1,'r',label='pmedia_bg')
    plt.plot(MU_energia,MU_intensidade,'k')
    #plt.axvline(x=643.52,color='k',linestyle='--',linewidth=0.5)
    plt.xlabel('Photon energy [eV]')
    plt.ylabel('Absorption coeficient, $\mu$ [m$^{-1}$]')
    plt.tight_layout()
    plt.legend(fontsize=12)
    
    plt.subplot(1,2,2)
    plt.title('Adjusted absorption coeficient',fontsize=10,weight='bold')
    plt.plot(energy,ca2,'r',label='nmedia_bg')
    plt.plot(MU_energia,MU_intensidade,'k')
    #plt.axvline(x=643.52,color='k',linestyle='--',linewidth=0.5)
    plt.xlabel('Photon energy [eV]')
    plt.ylabel('Absorption coeficient, $\mu$ [m$^{-1}$]')
    plt.tight_layout()
    plt.legend(fontsize=12)
    
    plt.savefig('dt5.png')
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Comparison between before and after correction',fontsize=9,weight='bold')
    plt.plot(energy, pmedia_bg/pmedia_bg[-1],'k.',ms=2,label='pmedia_bg')
    plt.plot(energy, pmedia_bg_corr/pmedia_bg_corr[-1],'b-',label='pmedia_bg corrected')
    plt.plot(energy, nmedia_bg/nmedia_bg[-1],'k*',ms=2,label='nmedia_bg')
    plt.plot(energy, nmedia_bg_corr/nmedia_bg_corr[-1],'r-',label='nmedia_bg corrected') 
    plt.plot(energy, dif,label='xmcd')
    plt.plot(energy, dif_corr,label='xmcd corrected')
    plt.xlabel('Photon energy [eV]')
    plt.ylabel('Normalized Intensity [a. u]')
    plt.tight_layout()
    plt.legend(fontsize=9)
    
    plt.subplot(1,2,2)
    plt.title('Comparison between before and after correction',fontsize=9,weight='bold')
    plt.plot(energy, xas/xas[-1],label='xas iso')
    plt.plot(energy, xas_corr/xas_corr[-1],label='xas iso corrected')
    plt.xlabel('Photon energy [eV]')
    plt.ylabel('Normalized Intensity [a. u]')
    plt.tight_layout()
    plt.legend(fontsize=10)
    
    plt.savefig('dt6.png')
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('XMCD comparison between before and after correction',fontsize=7,weight='bold')
    #plt.plot(energy, pmedia_bg_nor,label='pmedia_bg_nor')
    #plt.plot(energy, nmedia_bg_nor,label='nmedia_bg_nor')
    #plt.plot(energy, xas_nor,label='xas iso')
    plt.axhline(y=0,color='k',linestyle='--',linewidth=0.5)
    plt.plot(energy, dif_nor,label='xmcd')
    plt.plot(energy, dif_nor_corr,label='xmcd corrected')
    plt.xlabel('Photon energy [eV]')
    plt.ylabel('Intensity [a. u]')
    #plt.axis([630,670,-0.2,0.08])
    plt.tight_layout()
    plt.legend(fontsize=6)
    
    plt.subplot(1,2,2)
    plt.title('Normalized data after correction',fontsize=10,weight='bold')
    plt.plot(energy, pmedia_bg_nor_corr,label='pmedia_bg_nor corrected')
    plt.plot(energy, nmedia_bg_nor_corr,label='nmedia_bg_nor corrected') 
    plt.plot(energy, xas_nor_corr,label='xas iso corrected')
    plt.plot(energy, dif_nor_corr,label='xmcd corrected')
    plt.axhline(y=0,color='k',linestyle='--',linewidth=0.5)
    plt.xlabel('Photon energy [eV]')
    plt.ylabel('Intensity [a. u]')
    plt.tight_layout()
    plt.legend(fontsize=6)
    
    plt.savefig('dt7.png') 
    
def XAS_and_XMCD_export(dir,de,ate,ext='.dat',interp=0,pos=[1,4,5,8],neg=[2,3,6,7],exc=[],nor=779.7,bg=772,y=2,shift=[779.7,-0.5,0.5],displacement=[],L3=[],l_e=[],thickness=[],name=''): 
            
    plt.rcParams['figure.dpi']=150
    plt.rcParams['figure.figsize']=(6,4)
        
    p,n,energy = XASload(dir,de,ate,ext,interp,pos,neg,exc,y,shift) 
           
    pmedia = XASavg(p) 
    nmedia = XASavg(n)
                
    pmedia_bg = XASbg(pmedia,energy,bg)
    nmedia_bg = XASbg(nmedia,energy,bg)
    
    xas = (pmedia_bg + nmedia_bg)/2
    dif = pmedia_bg - nmedia_bg 
    
    pmedia_bg_nor = XASnor(pmedia_bg,energy,xas,nor)
    nmedia_bg_nor = XASnor(nmedia_bg,energy,xas,nor)    
    xas_nor = XASnor(xas,energy,xas,nor)
    dif_nor = XASnor(dif,energy,xas,nor)
    
    ### Applying the saturation effects correction in TEY detection
    
    ###### Attenuation length and absorption coeficient extracted from HENKE: importar antes da macro como "Henke.dat"
    Henke = np.loadtxt('raw/Henke.dat',delimiter=',')
    atl_energia = Henke[:,0]+displacement
    atl_intensidade = Henke[:,1]*10**(-6)
    
    mu_energia = atl_energia
    mu_intensidade = 1/atl_intensidade
        
    ###### interpolating in the range of data: 
    number_of_points = len(energy)
    interpolate = sp.interpolate.interp1d(mu_energia,mu_intensidade,fill_value="extrapolate")
    MU_energia = np.linspace(630,670,number_of_points) # range for Mn L2,3 edge: 630,670 Obs: mesmo range dos dados!!!
    MU_intensidade = interpolate(MU_energia)
    
    ###### Background Fitting:
    LinearFit1 = np.polyfit(energy[-80:-1],pmedia_bg[-80:-1],1) 
    H1 = LinearFit1[1]+LinearFit1[0]*energy;
    C = 0
    D = 1; 
    BG1 = C*energy + H1*(1-(1/(1+np.exp((energy-L3)/D))))
    
    LinearFit2 = np.polyfit(energy[-80:-1],nmedia_bg[-80:-1],1) 
    H2 = LinearFit2[1]+LinearFit2[0]*energy;
    C = 0
    D = 1; 
    BG2 = C*energy + H2*(1-(1/(1+np.exp((energy-L3)/D))))
        
    ###### multiplicative factor
    factor1 = MU_intensidade/(BG1+1)
    factor2 = MU_intensidade/(BG2+1)
    
    ###### Absorption coeficient for the data
    ca1 = (pmedia_bg+1)*factor1
    ca2 = (nmedia_bg+1)*factor2
    
    ###### Applying the correction for thin films:  
    
    T = thickness
    
    l_x1 = 1/((ca1))
    F = 1/(1+l_e/(l_x1))
    pmedia_corr = F * (1 - np.exp(-T*((1/l_e)+(1/l_x1))) ) * pmedia_bg
    
    l_x2 = 1/((ca2))
    F = 1/(1+l_e/(l_x2))
    nmedia_corr = F * (1 - np.exp(-T*((1/l_e)+(1/l_x2))) ) * nmedia_bg
    
    ###### Defining the corrected variables
    
    #pmedia = XASavg(p) 
    #nmedia = XASavg(n)
                
    pmedia_bg_corr = XASbg(pmedia_corr,energy,bg)
    nmedia_bg_corr = XASbg(nmedia_corr,energy,bg)
    
    xas_corr = (pmedia_bg_corr + nmedia_bg_corr)/2
    dif_corr = pmedia_bg_corr - nmedia_bg_corr 
    
    pmedia_bg_nor_corr = XASnor(pmedia_bg_corr,energy,xas_corr,nor)
    nmedia_bg_nor_corr = XASnor(nmedia_bg_corr,energy,xas_corr,nor)    
    xas_nor_corr = XASnor(xas_corr,energy,xas_corr,nor)
    dif_nor_corr = XASnor(dif_corr,energy,xas_corr,nor)
   
    filename = name+'_XAS_and_XMCD_export.dat'
    head = '#col0,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18\n#energy,pmedia,nmedia,pmedia_bg,nmedia_bg,xas,dif,pmedia_bg_nor,nmedia_bg_nor,xas_nor,dif_nor,pmedia_bg_corr,nmedia_bg_corr,xas_corr,dif_corr,pmedia_bg_nor_corr,nmedia_bg_nor_corr,xas_nor_corr,dif_nor_corr\n'   
    
    file = open(filename,'w')
    
    file.write(head)
    pmedia_bg_nor = XASnor(pmedia_bg,energy,xas,nor)
    Energy = energy #-displacement
    for i in range(len(Energy)):
        line = str(Energy[i])+','+str(pmedia[i])+','+str(nmedia[i])+','+str(pmedia_bg[i])+','+str(nmedia_bg[i])+','+str(xas[i])+','+str(dif[i])+','+str(pmedia_bg_nor[i])+','+str(nmedia_bg_nor[i])+','+str(xas_nor[i])+','+str(dif_nor[i])+','+str(pmedia_bg_corr[i])+','+str(nmedia_bg_corr[i])+','+str(xas_corr[i])+','+str(dif_corr[i])+','+str(pmedia_bg_nor_corr[i])+','+str(nmedia_bg_nor_corr[i])+','+str(xas_nor_corr[i])+','+str(dif_nor_corr[i])+'\n'
        file.write(line)
        
    file.close() 
    
def XLD_saturationeffectscorrection(dir,de,ate,ext='.dat',interp=0,parac=[1,4,5,8],perpc=[2,3,6,7],exc=[],nor=575,bg=522,y=2,shift=[779.72,-0.5,0.5],displacement=[],L3=[],l_e=[],thickness=[]):
      
    p,o,energy = XASload(dir,de,ate,ext,interp,parac,perpc,exc,y,shift)
       
    pmedia = XASavg(p) 
    omedia = XASavg(o)
        
    pmedia_bg = XASbg(pmedia,energy,bg)
    omedia_bg = XASbg(omedia,energy,bg)
        
    pmedia_bg_nor = XASnor(pmedia_bg,energy,pmedia_bg,nor)
    omedia_bg_nor = XASnor(omedia_bg,energy,omedia_bg,nor)  
        
    XLD = omedia_bg_nor - pmedia_bg_nor 
    
    ### Applying the saturation effects correction in TEY detection
    
    ###### Attenuation length and absorption coeficient extracted from HENKE: importar antes da macro como "Henke.dat"
    Henke = np.loadtxt('raw/Henke.dat',delimiter=',')
    atl_energia = Henke[:,0]+displacement
    atl_intensidade = Henke[:,1]*10**(-6)
    
    mu_energia = atl_energia
    mu_intensidade = 1/atl_intensidade
        
    ###### interpolating in the range of data: 
    number_of_points = len(energy)
    interpolate = sp.interpolate.interp1d(mu_energia,mu_intensidade,fill_value="extrapolate")
    MU_energia = np.linspace(632.002,673,number_of_points) # range for Mn L2,3 edge: 630,670 Obs: mesmo range dos dados!!!
    MU_intensidade = interpolate(MU_energia)
    
    ###### Background Fitting:
    LinearFit1 = np.polyfit(energy[-800:-1],pmedia_bg[-800:-1],1) 
    H1 = LinearFit1[1]+LinearFit1[0]*energy;
    C = 0
    D = 1; 
    BG1 = C*energy + H1*(1-(1/(1+np.exp((energy-L3)/D))))
    
    LinearFit2 = np.polyfit(energy[-800:-1],omedia_bg[-800:-1],1) 
    H2 = LinearFit2[1]+LinearFit2[0]*energy;
    C = 0
    D = 1; 
    BG2 = C*energy + H2*(1-(1/(1+np.exp((energy-L3)/D))))
        
    ###### multiplicative factor
    factor1 = MU_intensidade/(BG1+1)
    factor2 = MU_intensidade/(BG2+1)
    
    ###### Absorption coeficient for the data
    ca1 = (pmedia_bg+1)*factor1
    ca2 = (omedia_bg+1)*factor2
    
    ###### Applying the correction for thin films:  
    
    T = thickness
    
    l_x1 = 1/((ca1))
    F = 1/(1+l_e/(l_x1))
    pmedia_corr = F * (1 - np.exp(-T*((1/l_e)+(1/l_x1))) ) * pmedia_bg
    
    l_x2 = 1/((ca2))
    F = 1/(1+l_e/(l_x2))
    omedia_corr = F * (1 - np.exp(-T*((1/l_e)+(1/l_x2))) ) * omedia_bg
    
    ###### Defining the corrected variables
    
    #pmedia = XASavg(p) 
    #omedia = XASavg(n)
                
    pmedia_bg_corr = XASbg(pmedia_corr,energy,bg)
    omedia_bg_corr = XASbg(omedia_corr,energy,bg)
    
    #xas_corr = (pmedia_bg_corr + omedia_bg_corr)/2
    #xld_corr = omedia_bg_corr - pmedia_bg_corr 
    
    pmedia_bg_nor_corr = XASnor(pmedia_bg_corr,energy,pmedia_bg_corr,nor)
    omedia_bg_nor_corr = XASnor(omedia_bg_corr,energy,omedia_bg_corr,nor)    
    #xas_nor_corr = XASnor(xas_corr,energy,xas_corr,nor)
    xld_nor_corr = omedia_bg_nor_corr - pmedia_bg_nor_corr
    
    ##### plot
    plt.rcParams['figure.dpi']=120
    plt.rcParams['figure.figsize']=(8,4)
    
    plt.figure
    plt.subplot(1,2,1)
    plt.title('Attenuation Length',fontsize=10,weight='bold')
    plt.plot(atl_energia,atl_intensidade,label='From Henke')
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Attenuation length (m)')
    plt.legend(fontsize=6)
    
    plt.subplot(1,2,2)
    plt.title('Absorption coeficient',fontsize=10,weight='bold')
    plt.plot(mu_energia,mu_intensidade,label='From Henke')
    plt.plot(MU_energia,MU_intensidade,'k')
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Absorption coeficient, $\mu$ (m$^{-1}$)')
    plt.legend(fontsize=6)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Background fitting',fontsize=10,weight='bold')
    plt.plot(energy,pmedia_bg,'.',label='pmedia_bg')
    plt.plot(energy,BG1,label='Background')
    plt.plot(energy[-800:-1],H1[-800:-1],label='Linear fit')
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Intensity (a. u)')
    plt.legend(fontsize=6)
    
    plt.subplot(1,2,2)
    plt.title('Background fitting',fontsize=10,weight='bold')
    plt.plot(energy,omedia_bg,'.',label='omedia_bg')
    plt.plot(energy,BG2,label='Background')
    plt.plot(energy[-800:-1],H2[-800:-1],label='Linear fit')
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Intensity (a. u)')
    plt.legend(fontsize=6)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Determining the scale factor',fontsize=10,weight='bold')
    plt.plot(MU_energia,MU_intensidade,label='Henke')
    plt.plot(energy,factor1,label='Scale factor for pmedia_bg = Henke/Background')
    plt.plot(energy,(BG1+1)*10**6,label='Background*10$^6$')
    plt.plot(energy,factor1*(BG1+1),'--',label='Proof: scale factor*Background')
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Absorption coeficient, $\mu$ (m$^{-1}$)')
    plt.legend(fontsize=6)
    
    plt.subplot(1,2,2)
    plt.title('Determining the scale factor',fontsize=10,weight='bold')
    plt.plot(MU_energia,MU_intensidade,label='Henke')
    plt.plot(energy,factor2,label='Scale factor for omedia_bg = Henke/Background')
    plt.plot(energy,(BG2+1)*10**6,label='Background*10$^6$')
    plt.plot(energy,factor2*(BG2+1),'--',label='Proof: scale factor*Background')
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Absorption coeficient, $\mu$ (m$^{-1}$)')
    plt.legend(fontsize=6)
        
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Adjusted absorption coeficient',fontsize=8,weight='bold')
    plt.plot(energy,ca1,'r',label='pmedia_bg')
    plt.plot(MU_energia,MU_intensidade,'k')
    #plt.axvline(x=atl_energia,color='k',linestyle='--',linewidth=0.5)
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Absorption coeficient, $\mu$ (m$^{-1}$)')
    plt.legend(fontsize=6)
    
    plt.subplot(1,2,2)
    plt.title('Adjusted absorption coeficient',fontsize=8,weight='bold')
    plt.plot(energy,ca2,'r',label='omedia_bg')
    plt.plot(MU_energia,MU_intensidade,'k')
    #plt.axvline(x=644.1,color='k',linestyle='--',linewidth=0.5)
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Absorption coeficient, $\mu$ (m$^{-1}$)')
    plt.legend(fontsize=6)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Comparison between before and after correction',fontsize=7,weight='bold')
    #plt.plot(energy, omedia_bg/omedia_bg[-1],label='omedia_bg')
    #plt.plot(energy, omedia_bg_corr/omedia_bg_corr[-1],label='omedia_bg corrected')
    #plt.plot(energy, pmedia_bg/pmedia_bg[-1],label='pmedia_bg')
    #plt.plot(energy, pmedia_bg_corr/pmedia_bg_corr[-1],label='pmedia_bg corrected')
    plt.plot(energy, XLD,label='xld')
    plt.plot(energy, xld_nor_corr,label='xld corrected')
    plt.axhline(y=0,color='k',linestyle='--',linewidth=0.5)
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Normalized Intensity (a. u)')
    #plt.axis([630,672,-0.05,0.05])
    plt.legend(fontsize=6)
    
    plt.subplot(1,2,2)
    plt.title('Normalized data after correction',fontsize=10,weight='bold')
    plt.plot(energy, omedia_bg_nor_corr,label='omedia_bg_nor corrected') 
    plt.plot(energy, pmedia_bg_nor_corr,label='pmedia_bg_nor corrected')
    #plt.plot(energy, xas_nor_corr,label='xas iso corrected')
    plt.plot(energy, xld_nor_corr,label='xld corrected')
    plt.axhline(y=0,color='k',linestyle='--',linewidth=0.5)
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Intensity (a. u)')
    plt.legend(fontsize=6)
       
    #plt.plot(energy, pmedia_bg_nor, linestyle='-',linewidth=1.2,color='black',label='paralell to c')
    #plt.plot(energy, omedia_bg_nor, linestyle='-',linewidth=1.2,color='red',label='perpendicular to c')
    #plt.plot(energy, XLD, linestyle='-',linewidth=1.2,color='blue',label='XLD')
    #CONFIGplt(ylabel='xas, xld')   
   
        
def XLD_saturationeffectscorrection_export(dir,de,ate,ext='.dat',interp=0,parac=[1,4,5,8],perpc=[2,3,6,7],exc=[],nor=575,bg=522,y=2,shift=[779.72,-0.5,0.5],displacement=[],L3=[],l_e=[],thickness=[],name=''):
      
    p,o,energy = XASload(dir,de,ate,ext,interp,parac,perpc,exc,y,shift)
       
    pmedia = XASavg(p) 
    omedia = XASavg(o)
        
    pmedia_bg = XASbg(pmedia,energy,bg)
    omedia_bg = XASbg(omedia,energy,bg)
        
    pmedia_bg_nor = XASnor(pmedia_bg,energy,pmedia_bg,nor)
    omedia_bg_nor = XASnor(omedia_bg,energy,omedia_bg,nor)  
        
    XLD = omedia_bg_nor - pmedia_bg_nor
    
    ### Applying the saturation effects correction in TEY detection
    
    ###### Attenuation length and absorption coeficient extracted from HENKE: importar antes da macro como "Henke.dat"
    Henke = np.loadtxt('raw/Henke.dat',delimiter=',')
    atl_energia = Henke[:,0]+displacement
    atl_intensidade = Henke[:,1]*10**(-6)
    
    mu_energia = atl_energia
    mu_intensidade = 1/atl_intensidade
        
    ###### interpolating in the range of data: 
    number_of_points = len(energy)
    interpolate = sp.interpolate.interp1d(mu_energia,mu_intensidade,fill_value="extrapolate")
    MU_energia = np.linspace(632.002,673,number_of_points) # range for Mn L2,3 edge: 630,670 Obs: mesmo range dos dados!!!
    MU_intensidade = interpolate(MU_energia)
    
    ###### Background Fitting:
    LinearFit1 = np.polyfit(energy[-800:-1],pmedia_bg[-800:-1],1) 
    H1 = LinearFit1[1]+LinearFit1[0]*energy;
    C = 0
    D = 1; 
    BG1 = C*energy + H1*(1-(1/(1+np.exp((energy-L3)/D))))
    
    LinearFit2 = np.polyfit(energy[-800:-1],omedia_bg[-800:-1],1) 
    H2 = LinearFit2[1]+LinearFit2[0]*energy;
    C = 0
    D = 1; 
    BG2 = C*energy + H2*(1-(1/(1+np.exp((energy-L3)/D))))
        
    ###### multiplicative factor
    factor1 = MU_intensidade/(BG1+1)
    factor2 = MU_intensidade/(BG2+1)
    
    ###### Absorption coeficient for the data
    ca1 = (pmedia_bg+1)*factor1
    ca2 = (omedia_bg+1)*factor2
    
    ###### Applying the correction for thin films:  
    
    T = thickness
    
    l_x1 = 1/((ca1))
    F = 1/(1+l_e/(l_x1))
    pmedia_corr = F * (1 - np.exp(-T*((1/l_e)+(1/l_x1))) ) * pmedia_bg
    
    l_x2 = 1/((ca2))
    F = 1/(1+l_e/(l_x2))
    omedia_corr = F * (1 - np.exp(-T*((1/l_e)+(1/l_x2))) ) * omedia_bg
    
    ###### Defining the corrected variables
    
    #pmedia = XASavg(p) 
    #omedia = XASavg(n)
                
    pmedia_bg_corr = XASbg(pmedia_corr,energy,bg)
    omedia_bg_corr = XASbg(omedia_corr,energy,bg)
    
    #xas_corr = (pmedia_bg_corr + omedia_bg_corr)/2
    #xld_corr = omedia_bg_corr - pmedia_bg_corr 
    
    pmedia_bg_nor_corr = XASnor(pmedia_bg_corr,energy,pmedia_bg_corr,nor)
    omedia_bg_nor_corr = XASnor(omedia_bg_corr,energy,omedia_bg_corr,nor)    
    #xas_nor_corr = XASnor(xas_corr,energy,xas_corr,nor)
    xld_nor_corr = omedia_bg_nor_corr - pmedia_bg_nor_corr
    
    filename = name+'_XLD_export.dat'
    head = '#col0,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12\n#energy,pmedia,omedia,pmedia_bg,omedia_bg,pmedia_bg_nor,omedia_bg_nor,xld,pmedia_bg_corr,omedia_bg_corr,pmedia_bg_nor_corr,omedia_bg_nor_corr,xld_corr\n'   
    
    file = open(filename,'w')
    file.write(head)
    
    Energy = energy #-displacement
    
    for i in range(len(Energy)):
        line = str(Energy[i])+','+str(pmedia[i])+','+str(omedia[i])+','+str(pmedia_bg[i])+','+str(omedia_bg[i])+','+str(pmedia_bg_nor[i])+','+str(omedia_bg_nor[i])+','+str(XLD[i])+','+str(pmedia_bg_corr[i])+','+str(omedia_bg_corr[i])+','+str(pmedia_bg_nor_corr[i])+','+str(omedia_bg_nor_corr[i])+','+str(xld_nor_corr[i])+'\n'
        file.write(line)
        
    file.close() 
    
    plt.rcParams['figure.dpi']=150
    plt.rcParams['figure.figsize']=(6,4)
    
def XLDplot2(dir,de,ate,ext='.dat',interp=0,parac=[1,4,5,8],perpc=[2,3,6,7],exc=[],nor=575,bg=522,y=2,shift=[779.72,-0.5,0.5]):
      
    p,o,energy = XASload(dir,de,ate,ext,interp,parac,perpc,exc,y,shift)
       
    pmedia = XASavg(p) 
    omedia = XASavg(o)
        
    pmedia_bg = XASbg(pmedia,energy,bg)
    omedia_bg = XASbg(omedia,energy,bg)
        
    pmedia_bg_nor = XASnor(pmedia_bg,energy,pmedia_bg,nor)
    omedia_bg_nor = XASnor(omedia_bg,energy,omedia_bg,nor)  
        
    XLD = -(pmedia_bg_nor - omedia_bg_nor)
    
    plt.rcParams['figure.dpi']=150
    plt.rcParams['figure.figsize']=(6,4)
    
    plt.figure()
    plt.plot(energy, pmedia_bg_nor, linestyle='-',linewidth=1.2,color='black',label='$\parallel$c')
    plt.plot(energy, omedia_bg_nor, linestyle='-',linewidth=1.2,color='red',label='$\perp$c')
    plt.plot(energy, XLD, linestyle='-',linewidth=1.2,color='blue',label='XLD')
    plt.axhline(y=0,color='k',linestyle='--',linewidth=0.5)
    plt.axvline(x=nor,color='k',linestyle='--',linewidth=0.5)
    plt.grid(False)
    
    
    CONFIGplt(ylabel='xas, xld')   
    
#######################################################
#######################################################
    ###    functions that I didn't used:
#######################################################
#######################################################
def XASplotBG(dir,scans=[],ext='.dat',marker='',y=2,shift=[779.72,-0.5,0.5],ymin=888,ymax=888,bg=775): 
             
    for i in scans:
        
        c = '{:04d}'.format(i)
            
        if '2' in ext:
            energy,col2,col3,col4,i0,sample,norm,col8,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=1,delimiter=',',unpack=True)
        else:
            col0,energy,col2,col3,col4,col5,i0,col7,sample,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=7,delimiter=',',unpack=True)
            norm = np.array(sample/i0)
        
        if y == 0:    yaxis = i0;   ylab = 'i0';    xaxis = energy
        elif y == 1:  yaxis = sample;   ylab = 'sample';    xaxis = energy
        elif y == 2:    yaxis = norm;   ylab = 'norm';    xaxis = energy
        elif y == 11:
            energy_shift,energy_cut,sample_cut,popt,amp,x0,sigma,gauss = XASshift(energy,sample,ref=shift[0],left=shift[1],right=shift[2])
            yaxis = sample;   ylab = 'sample';    xaxis = energy_shift 
        elif y == 21:
            energy_shift,energy_cut,norm_cut,popt,amp,x0,sigma,gauss = XASshift(energy,norm,ref=shift[0],left=shift[1],right=shift[2])
            yaxis = norm;   ylab = 'norm';    xaxis = energy_shift 
        
        yaxis_bg = XASbg(yaxis,xaxis,bg)    
        plt.plot(xaxis, yaxis_bg, linestyle='-',linewidth=1.2, label=str(i), marker=marker)
        
    if ymin!= 888:
        plt.ylim((ymin,ymax))    
        CONFIGplt(ylabel=ylab)              
    else:
        CONFIGplt(ylabel=ylab)  
    
def XASplotXMCD(dir,de,ate,ext='.dat',marker='',pos=[1,4,5,8],neg=[2,3,6,7],exc=[],y=2,shift=[779.72,-0.5,0.5]):    
    
    for i in range(de,ate+1):
            
        c = '{:04d}'.format(i)
        
        if '2' in ext:
            energy,col2,col3,col4,i0,sample,norm,col8,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=1,delimiter=',',unpack=True)
        else:
            col0,energy,col2,col3,col4,col5,i0,col7,sample,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=7,delimiter=',',unpack=True)
            norm = np.array(sample/i0)
            
        if y == 0:    yaxis = i0;   ylab = 'i0';    xaxis = energy
        elif y == 1:  yaxis = sample;   ylab = 'sample';    xaxis = energy
        elif y == 2:    yaxis = norm;   ylab = 'norm';    xaxis = energy
        elif y == 11:
            energy_shift,energy_cut,sample_cut,popt,amp,x0,sigma,gauss = XASshift(energy,sample,ref=shift[0],left=shift[1],right=shift[2])
            yaxis = sample;   ylab = 'sample';    xaxis = energy_shift 
        elif y == 21:
            energy_shift,energy_cut,norm_cut,popt,amp,x0,sigma,gauss = XASshift(energy,norm,ref=shift[0],left=shift[1],right=shift[2])
            yaxis = norm;   ylab = 'norm';    xaxis = energy_shift  
                
        c = int(c)
        t = c - de + 1      
        alabel = str(t)+' - '+str(i)
            
        if (t in pos) and (t not in exc):    
            plt.plot(xaxis, yaxis, linestyle='-', color='black', linewidth=1.2, label=alabel, marker=marker)
        elif (t in neg) and (t not in exc):    
            plt.plot(xaxis, yaxis, linestyle='-', color='red', linewidth=1.2, label=alabel, marker=marker)
    
    CONFIGplt(ylabel=ylab) 

def XMCDplot(dir,de,ate,ext='.dat',interp=0,pos=[1,4,5,8],neg=[2,3,6,7],exc=[],nor=779.7,bg=772,y=2,shift=[779.7,-0.5,0.5]): 
             
    p,n,energy = XASload(dir,de,ate,ext,interp,pos,neg,exc,y,shift)     
           
    pmedia = XASavg(p) 
    nmedia = XASavg(n)
                
    pmedia_bg = XASbg(pmedia,energy,bg)
    nmedia_bg = XASbg(nmedia,energy,bg)
    
    xas = (pmedia_bg + nmedia_bg)/2
    dif = pmedia_bg - nmedia_bg 
    
    pmedia_bg_nor = XASnor(pmedia_bg,energy,xas,nor)
    nmedia_bg_nor = XASnor(nmedia_bg,energy,xas,nor)    
    xas_nor = XASnor(xas,energy,xas,nor)
    dif_nor = XASnor(dif,energy,xas,nor)
   
    plt.plot(energy, pmedia_bg_nor,'*-',color='black',ms=1,linewidth=0.8, label='pos_avg'); 
    plt.plot(energy, nmedia_bg_nor,'*-', color='red',ms=1,linewidth=0.8, label='neg_avg'); 
    #plt.plot(energy, xas_nor, linestyle='-', color='green',linewidth=0.5, label='xas_avg'); 
    plt.plot(energy, dif_nor, linestyle='-', color='blue',linewidth=1.2, label='xmcd_asymm'); 
     
    CONFIGplt(ylabel='xas, xmcd asymmetry')  
       
def XMCDexport(dir,de,ate,ext='.dat',interp=0,pos=[1,4,5,8],neg=[2,3,6,7],exc=[],nor=779.7,bg=772,y=2,shift=[779.7,-0.5,0.5]): 
            
    p,n,energy = XASload(dir,de,ate,ext,interp,pos,neg,exc,y,shift) 
           
    pmedia = XASavg(p) 
    nmedia = XASavg(n)
                
    pmedia_bg = XASbg(pmedia,energy,bg)
    nmedia_bg = XASbg(nmedia,energy,bg)
    
    xas = (pmedia_bg + nmedia_bg)/2
    dif = pmedia_bg - nmedia_bg 
    
    pmedia_bg_nor = XASnor(pmedia_bg,energy,xas,nor)
    nmedia_bg_nor = XASnor(nmedia_bg,energy,xas,nor)    
    xas_nor = XASnor(xas,energy,xas,nor)
    dif_nor = XASnor(dif,energy,xas,nor)
   
    filename = dir+str(de)+'-'+str(ate)+'-XMCDexport.dat'
    head = '#energy,pmedia,nmedia,pmedia_bg,nmedia_bg,xas,dif,pmedia_bg_nor,nmedia_bg_nor,xas_nor,dif_nor\n'   
    
    file = open(filename,'w')
    
    file.write(head)
    pmedia_bg_nor = XASnor(pmedia_bg,energy,xas,nor)
    for i in range(len(energy)):
        line = str(energy[i])+','+str(pmedia[i])+','+str(nmedia[i])+','+str(pmedia_bg[i])+','+str(nmedia_bg[i])+','+str(xas[i])+','+str(dif[i])+','+str(pmedia_bg_nor[i])+','+str(nmedia_bg_nor[i])+','+str(xas_nor[i])+','+str(dif_nor[i])+'\n'
        file.write(line)
        
    file.close() 

def XMCDplot2(dir,de,ate,ext='.dat',interp=0,pos=[1,4,5,8],neg=[2,3,6,7],exc=[],nor=779.7,bg=772,y=2,shift=[779.7,-0.5,0.5]): 
          
    p,n,energy = XASload(dir,de,ate,ext,interp,pos,neg,exc,y,shift)
    
    p_bg = []
    n_bg = []       
    for i in p:
        v = XASbg(i,energy,bg)
        p_bg.append(v)        
    for i in n:
        v = XASbg(i,energy,bg)
        n_bg.append(v)
            
    p_bg_media = XASavg(p_bg) 
    n_bg_media = XASavg(n_bg)
    
    xas = (p_bg_media + n_bg_media)/2
    dif = p_bg_media - n_bg_media 
    
    p_bg_media_nor = XASnor(p_bg_media,energy,xas,nor)
    n_bg_media_nor = XASnor(n_bg_media,energy,xas,nor)    
    xas_nor = XASnor(xas,energy,xas,nor)
    dif_nor = XASnor(dif,energy,xas,nor)
   
    plt.plot(energy, p_bg_media_nor, linestyle='-', color='black',linewidth=1.5, label='pos_avg'); 
    plt.plot(energy, n_bg_media_nor, linestyle='-', color='red',linewidth=1.5, label='neg_avg'); 
    plt.plot(energy, xas_nor, linestyle='-', color='green',linewidth=0.5, label='xas_avg'); 
    plt.plot(energy, dif_nor, linestyle='-', color='blue',linewidth=1.5, label='xmcd_asymm'); 
     
    CONFIGplt(ylabel='xas, xmcd asymmetry')
    
    
def XMCDexport2(dir,de,ate,ext='.dat',interp=0,pos=[1,4,5,8],neg=[2,3,6,7],exc=[],nor=779.7,bg=772,y=2,shift=[779.7,-0.5,0.5]): 
            
    p,n,energy = XASload(dir,de,ate,ext,interp,pos,neg,exc,y,shift) 
           
    p_bg = []
    n_bg = []       
    for i in p:
        v = XASbg(i,energy,bg)
        p_bg.append(v)        
    for i in n:
        v = XASbg(i,energy,bg)
        n_bg.append(v)
            
    p_bg_media = XASavg(p_bg) 
    n_bg_media = XASavg(n_bg)
    
    xas = (p_bg_media + n_bg_media)/2
    dif = p_bg_media - n_bg_media 
    
    p_bg_media_nor = XASnor(p_bg_media,energy,xas,nor)
    n_bg_media_nor = XASnor(n_bg_media,energy,xas,nor)    
    xas_nor = XASnor(xas,energy,xas,nor)
    dif_nor = XASnor(dif,energy,xas,nor)
   
    filename = dir+str(de)+'-'+str(ate)+'-XMCDexport.dat'
    head = '#energy,pmedia,nmedia,pmedia_bg,nmedia_bg,xas,dif,pmedia_bg_nor,nmedia_bg_nor,xas_nor,dif_nor\n'   
    
    file = open(filename,'w')
    
    file.write(head)
    
    for i in range(len(energy)):
        line = str(energy[i])+','+str(p_bg[i])+','+str(n_bg[i])+','+str(p_bg_media[i])+','+str(n_bg_media[i])+','+str(xas[i])+','+str(dif[i])+','+str(p_bg_media_nor[i])+','+str(n_bg_media_nor[i])+','+str(xas_nor[i])+','+str(dif_nor[i])+'\n'
        file.write(line)
        
    file.close()         


def XMCDintegrate(dir,de,ate,ext='.dat',interp=0,pos=[1,4,5,8],neg=[2,3,6,7],exc=[],nor=779.7,bg=772,y=2,shift=[779.7,-0.5,0.5]):
    
#    XMCDexport(dir,de,ate,ext='.dat',interp=0,pos=[1,4,5,8],neg=[2,3,6,7],exc=[],nor=779.7,bg=772,y=2,shift=[779.7,-0.5,0.5])
    XMCDexport(dir,de,ate,ext=ext,interp=interp,pos=pos,neg=neg,exc=exc,nor=nor,bg=bg,y=y,shift=shift)  
    filename = dir+str(de)+'-'+str(ate)+'-XMCDexport.dat'
    energy,pmedia,nmedia,pmedia_bg,nmedia_bg,xas,dif,pmedia_bg_nor,nmedia_bg_nor,xas_nor,dif_nor = np.loadtxt(filename,skiprows=1,delimiter=',',unpack=True)
        
    #procurando o índice dos arrays para o valor de energia background  
    teste1 = []    
    for v in energy:
        b = abs(bg - v)
        teste1.append(round(b,1))
    n1 = teste1.index(min(teste1))
    
    #integrando a partir do índice encontrado antes
    energy_cut = energy[n1:]
    dif_nor_cut = dif_nor[n1:]
    y_int = integrate.cumtrapz(dif_nor_cut, energy_cut, initial=0)
        
    #plotando as integrais
    
    plt.plot(energy_cut, dif_nor_cut, 'bo', label = 'xmcd asymmetry')
    plt.plot(energy_cut, y_int, 'r-', label = 'integrate')
    
    CONFIGplt(ylabel='xmcd asymmetry, integrate')
    
    
def XMCDintegrateexport(dir,de,ate,ext='.dat',interp=0,pos=[1,4,5,8],neg=[2,3,6,7],exc=[],nor=779.7,bg=772,y=2,shift=[779.7,-0.5,0.5]):
    
#    XMCDexport(dir,de,ate,ext='.dat',interp=0,pos=[1,4,5,8],neg=[2,3,6,7],exc=[],nor=779.7,bg=772,y=2,shift=[779.7,-0.5,0.5])
    XMCDexport(dir,de,ate,ext=ext,interp=interp,pos=pos,neg=neg,exc=exc,nor=nor,bg=bg,y=y,shift=shift)  
    filename = dir+str(de)+'-'+str(ate)+'-XMCDexport.dat'
    energy,pmedia,nmedia,pmedia_bg,nmedia_bg,xas,dif,pmedia_bg_nor,nmedia_bg_nor,xas_nor,dif_nor = np.loadtxt(filename,skiprows=1,delimiter=',',unpack=True)
        
    #procurando o índice dos arrays para o valor de energia background  
    teste1 = []    
    for v in energy:
        b = abs(bg - v)
        teste1.append(round(b,1))
    n1 = teste1.index(min(teste1))
    
    #integrando a partir do índice encontrado antes
    energy_cut = energy[n1:]
    dif_nor_cut = dif_nor[n1:]
    y_int = integrate.cumtrapz(dif_nor_cut, energy_cut, initial=0)
        
    #escrevendo o arquivo novo
    
    head = '#energy,dif_nor,y_int\n'   
    
    fname = dir+str(de)+'-'+str(ate)+'-XMCDintegrate.dat'
    file = open(fname,'w')
    
    file.write(head)
    
    for i in range(len(energy_cut)):
        line = str(energy_cut[i])+','+str(dif_nor_cut[i])+','+str(y_int[i])+'\n'
        file.write(line)
        
    file.close()

def XLDplot(dir,de,ate,ext='.dat',interp=0,parac=[1,4,5,8],perpc=[2,3,6,7],exc=[],nor=575,bg=522,y=2,shift=[779.72,-0.5,0.5]):
      
    p,o,energy = XASload(dir,de,ate,ext,interp,parac,perpc,exc,y,shift)
       
    pmedia = XASavg(p) 
    omedia = XASavg(o)
        
    pmedia_bg = XASbg(pmedia,energy,bg)
    omedia_bg = XASbg(omedia,energy,bg)
        
    pmedia_bg_nor = XASnor(pmedia_bg,energy,pmedia_bg,nor)
    omedia_bg_nor = XASnor(omedia_bg,energy,omedia_bg,nor)  
        
    XLD = pmedia_bg_nor - omedia_bg_nor
    
    plt.plot(energy, pmedia_bg_nor, linestyle='-',linewidth=1.2,color='black',label='paralell to c')
    plt.plot(energy, omedia_bg_nor, linestyle='-',linewidth=1.2,color='red',label='perpendicular to c')
    plt.plot(energy, XLD, linestyle='-',linewidth=1.2,color='blue',label='XLD')
    
    CONFIGplt(ylabel='xas, xld')   
   
        
def XLDexport(dir,de,ate,ext='.dat',interp=0,parac=[1,4,5,8],perpc=[2,3,6,7],exc=[],nor=575,bg=522,y=2,shift=[779.72,-0.5,0.5]):
      
    p,o,energy = XASload(dir,de,ate,ext,interp,parac,perpc,exc,y,shift)
       
    pmedia = XASavg(p) 
    omedia = XASavg(o)
        
    pmedia_bg = XASbg(pmedia,energy,bg)
    omedia_bg = XASbg(omedia,energy,bg)
        
    pmedia_bg_nor = XASnor(pmedia_bg,energy,pmedia_bg,nor)
    omedia_bg_nor = XASnor(omedia_bg,energy,omedia_bg,nor)  
        
    XLD = pmedia_bg_nor - omedia_bg_nor
    
    filename = dir+str(de)+'-'+str(ate)+'-XLDexport.dat'
    head = '#energy,pmedia,omedia,pmedia_bg,omedia_bg,pmedia_bg_nor,omedia_bg_nor,XLD\n'   
    
    file = open(filename,'w')
    
    file.write(head)
    
    for i in range(len(energy)):
        line = str(energy[i])+','+str(pmedia[i])+','+str(omedia[i])+','+str(pmedia_bg[i])+','+str(omedia_bg[i])+','+str(pmedia_bg_nor[i])+','+str(omedia_bg_nor[i])+','+str(XLD[i])+'\n'
        file.write(line)
        
    file.close()          


def XLDplot2(dir,de,ate,ext='.dat',interp=0,parac=[1,4,5,8],perpc=[2,3,6,7],exc=[],nor=779.7,bg=522,corr=0.91,y=2,shift=[779.72,-0.5,0.5]):
      
    p,o,energy = XASload(dir,de,ate,ext,interp,parac,perpc,exc,y,shift)
       
    pmedia = XASavg(p) 
    omedia = XASavg(o)
    
    pmedia_corr = pmedia/corr
    
    pmedia_corr_bg = XASbg(pmedia_corr,energy,bg)
    omedia_bg = XASbg(omedia,energy,bg)
    
    XLD = pmedia_corr_bg - omedia_bg
    
    plt.plot(energy, pmedia_corr_bg, linestyle='-',linewidth=1.2,color='black',label='paralell to c')
    plt.plot(energy, omedia_bg, linestyle='-',linewidth=1.2,color='red',label='perpendicular to c')
    plt.plot(energy, XLD, linestyle='-',linewidth=1.2,color='blue',label='XLD')
    
    CONFIGplt(ylabel='xas, xld')   

    
def PROFplot(dir,yaw=[-3949,-4030,-4148,-4236,-4346],ext='.dat',xmin=1,xmax=1,ymin=1,ymax=1,zmin=1,zmax=1,levels=100,color='rainbow'):
    
    #importing the data and defining arrays    
    
    z = []
    norm = []
    
    for i in yaw:
        
        #z_i,i0_i,sample_i,col4,col5,norm_i = np.loadtxt(dir+str(i)+ext,skiprows=8,delimiter=' ',unpack=True,dtype={'names':('z','i0','sample','col4','col5','norm'),'formats':(np.float,np.float,np.float,np.unicode_,np.unicode_,np.float)})
        z_i,norm_i = np.loadtxt(dir+str(i)+ext,usecols=(0,5),skiprows=8,delimiter=' ',unpack=True)
        z.append(z_i)
        norm.append(norm_i)
            
    #building the 2D grid
    
    z = np.resize(z,(1,len(yaw)*len(z_i)))
    z = z[0]
          
    r = len(z)//len(yaw)
    k = len(yaw) 
    
    #defining new arrays with the correct shape (rows,columns)   
    
    yaw_temp = []
    for v in range(k):
        for i in range(r):
            yaw_temp.append(yaw[v])

    yaw = yaw_temp
    
    norm = np.resize(norm,(1,k*r))
    norm = norm[0]
   
    z = np.reshape(z,(k,r))
    yaw = np.reshape(yaw,(k,r))
    norm = np.reshape(norm,(k,r))
    
    #defininf x and y axis for the plot

    if xmin==1:
        x_i = np.amin(yaw)
    else:
        x_i = xmin
        
    if xmax==1:
        x_f = np.amax(yaw)
    else:
        x_f = xmax
        
    if ymin==1:
        y_i = np.amin(z)
    else:
        y_i = ymin
        
    if ymax==1:
        y_f = np.amax(z)
    else:
        y_f = ymax
   
    #plotting the data
    
    plt.contourf(yaw, z, norm, int(levels), cmap=color)
    
    plt.title(dir, fontsize=13)
    plt.ylabel('z (mm)')
    plt.xlabel('yaw (microrad)')
    plt.axis([x_i,x_f,y_i,y_f])
    plt.grid(True)
    
    plt.colorbar(label='counts', ticks=[])
      

def PROFplot2(dir,de,ate,yaw=[-3949,-4030,-4148,-4236,-4346],ext='.dat',xmin=1,xmax=1,ymin=1,ymax=1,zmin=1,zmax=1,levels=100,color='rainbow'):
    
    #importing the data and defining arrays    
    
    z = []
    norm = []
    
    for i in range(de,ate+1):
            
        c = '{:04d}'.format(i)
        z_i,i0_i,sample_i,col4,col5,norm_i = np.loadtxt(dir+'_'+c+ext,skiprows=8,delimiter=' ',unpack=True,dtype={'names':('z','i0','sample','col4','col5','norm'),'formats':(np.float,np.float,np.float,np.string_,np.string_,np.float)})
        z.append(z_i)
        norm.append(norm_i)
            
    #building the 2D grid
    
    z = np.resize(z,(1,len(yaw)*len(z_i)))
    z = z[0]
          
    r = len(z)//len(yaw)
    k = len(yaw) 
    
    #defining new arrays with the correct shape (rows,columns)   
    
    yaw_temp = []
    for v in range(k):
        for i in range(r):
            yaw_temp.append(yaw[v])

    yaw = yaw_temp
    
    norm = np.resize(norm,(1,k*r))
    norm = norm[0]
   
    z = np.reshape(z,(k,r))
    yaw = np.reshape(yaw,(k,r))
    norm = np.reshape(norm,(k,r))
    
    #defininf x and y axis for the plot

    if xmin==1:
        x_i = np.amin(yaw)
    else:
        x_i = xmin
        
    if xmax==1:
        x_f = np.amax(yaw)
    else:
        x_f = xmax
        
    if ymin==1:
        y_i = np.amin(z)
    else:
        y_i = ymin
        
    if ymax==1:
        y_f = np.amax(z)
    else:
        y_f = ymax
   
    #plotting the data
    
    plt.contourf(yaw, z, norm, int(levels), cmap=color)
    
    plt.title('quadradinho', fontsize=13)
    plt.ylabel('z (mm)')
    plt.xlabel('yaw (microrad)')
    plt.axis([x_i,x_f,y_i,y_f])
    plt.grid(True)
    
    plt.colorbar(label='counts', ticks=[])

def PROFXplot(dir,x=[29.1,29.2,29.3,29.4,29.5],ext='.dat',xmin=1,xmax=1,ymin=1,ymax=1,zmin=1,zmax=1,levels=100,color='rainbow'):
    
    #importing the data and defining arrays    
    
    z = []
    norm = []
    
    for i in x:
        
        #z_i,i0_i,sample_i,col4,col5,norm_i = np.loadtxt(dir+str(i)+ext,skiprows=8,delimiter=' ',unpack=True,dtype={'names':('z','i0','sample','col4','col5','norm'),'formats':(np.float,np.float,np.float,np.unicode_,np.unicode_,np.float)})
        z_i,norm_i = np.loadtxt(dir+str(i)+ext,usecols=(0,5),skiprows=8,delimiter=' ',unpack=True)
        z.append(z_i)
        norm.append(norm_i)
            
    #building the 2D grid
    
    z = np.resize(z,(1,len(x)*len(z_i)))
    z = z[0]
          
    r = len(z)//len(x)
    k = len(x) 
    
    #defining new arrays with the correct shape (rows,columns)   
    
    x_temp = []
    for v in range(k):
        for i in range(r):
            x_temp.append(x[v])

    x = x_temp
    
    norm = np.resize(norm,(1,k*r))
    norm = norm[0]
   
    z = np.reshape(z,(k,r))
    x = np.reshape(x,(k,r))
    norm = np.reshape(norm,(k,r))
    
    #defining x and y axis for the plot

    if xmin==1:
        x_i = np.amin(x)
    else:
        x_i = xmin
        
    if xmax==1:
        x_f = np.amax(x)
    else:
        x_f = xmax
        
    if ymin==1:
        y_i = np.amin(z)
    else:
        y_i = ymin
        
    if ymax==1:
        y_f = np.amax(z)
    else:
        y_f = ymax
   
    #plotting the data
    
    plt.contourf(x, z, norm, int(levels), cmap=color)
    
    plt.title(dir, fontsize=13)
    plt.ylabel('z (mm)')
    plt.xlabel('x (mm)')
    plt.axis([x_i,x_f,y_i,y_f])
    plt.grid(True)
    
    plt.colorbar(label='counts', ticks=[])

def Zplot(dir,scans,y=2):
   
    for i in scans:
        
        c = '{:04d}'.format(i)
        
        z,i0,isam,nor=np.loadtxt(dir+'_'+c+'.dat',skiprows=6,delimiter=' ',unpack=True,usecols=(0,1,2,5))
    
        if y == 0:    yaxis = i0;   ylab = 'i0';    xaxis = z
        elif y == 1:  yaxis = isam;   ylab = 'sample';    xaxis = z
        elif y == 2:  yaxis = nor;   ylab = 'norm';    xaxis = z
        
        elif y==-1: 
            if i==scans[0]: yaxis_1 = isam;   ylab = 'sample'; xaxis = z
            else: yaxis_2 = isam;   ylab = 'sample';    xaxis = z
        elif y==-2: 
            if i==scans[0]: yaxis_1 = nor;   ylab = 'sample'; xaxis = z
            else: yaxis_2 = nor;   ylab = 'sample';    xaxis = z
        
    if y==-1 or y==-2:
      plt.plot(xaxis, yaxis_1, linestyle='-',linewidth=1.2, color='black',label='at edge')
      plt.plot(xaxis, yaxis_2, linestyle='-',linewidth=1.2, color='red',label='out edge')
      yaxis=yaxis_1-yaxis_2 
                      
    plt.plot(xaxis, yaxis, linestyle='-',linewidth=1.2, color='blue',label='dif')            
            
    CONFIGplt(ylabel=ylab,xlabel='z',grid=True)
    
    
def MESHplot(dir,xmotor='bercox',ymotor='mbobz',xmin=1,xmax=1,ymin=1,ymax=1,zmin=1,zmax=1,levels=500,log=0,color='rainbow'):
       
    #importing the data and defining arrays
    
    x,y,i0,isam=np.loadtxt(dir,skiprows=8,delimiter=' ',unpack=True,usecols=(0,1,2,3))
    norm=isam/i0
       
    #finding out the number of rows and columns for the grid
    
    t = np.array(y)
    i = 0
    vi = t[i]
    
    for i, v in enumerate(t):   
        if v==vi:
            if i>2:
                r=i
                break
        else:
            i+=1
            
    c = len(x)//(r) #'//' em vez de '/' é para resultar num inteiro
    
    #defining new arrays with the correct shape (rows,columns)
    
    x = np.reshape(x,(c,r))
    y = np.reshape(y,(c,r))
    norm = np.reshape(norm,(c,r))
    
    #defining the levels' scale for the plot
    
    if log==1:
        
        z_i = np.log10(zmin)
        
        if zmax==1:
            z_f = np.log10(np.amax(norm))
        else:
            z_f = np.log10(zmax)
        
        nl = levels 
        
        alevels = np.logspace(z_i,z_f,nl,base=10)
        
    else:
        
        if zmin==1:
            z_i = np.amin(norm)
        else:
            z_i = zmin
        
        if xmax==1:
            z_f = np.amax(norm)
        else:
            z_f = zmax
        
        nl = levels 
        
        alevels = np.linspace(z_i,z_f,nl)
          
    #defininf x and y axis for the plot

    if xmin==1:
        x_i = np.amin(x)
    else:
        x_i = xmin
        
    if xmax==1:
        x_f = np.amax(x)
    else:
        x_f = xmax
        
    if ymin==1:
        y_i = np.amin(y)
    else:
        y_i = ymin
        
    if ymax==1:
        y_f = np.amax(y)
    else:
        y_f = ymax
   
    #plotting the data
    
    if log==1:
        plt.contourf(x,y,norm,locator=ticker.LogLocator(),levels=alevels,cmap=color)
    else:
        plt.contourf(x,y,norm,locator=ticker.LinearLocator(),levels=alevels,cmap=color)
    
    plt.title(dir, fontsize=13)
    plt.ylabel(ymotor)
    plt.xlabel(xmotor)
    plt.axis([x_i,x_f,y_i,y_f])
    plt.grid(True)
    
    plt.colorbar(label='TEY', ticks=[])
