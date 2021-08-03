import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import integrate
from matplotlib import ticker

#%pylab
#%matplotlib inline


def XASload(dir,de,ate,ext,interp,groupA,groupB,exc,y,shift):    #ex: p,n,energy = XASload(dir,de,ate,ext,y,pos,neg,exc)
          
    a=[];b=[];
  
    for i in range(de,ate+1):
            
        c = '{:04d}'.format(i)
        
        if '2' in ext:
            energy_i,col2,col3,col4,i0_i,sample_i,norm_i,col8,col9 = np.loadtxt(dir+'_'+c+ext,skiprows=1,delimiter=',',unpack=True)
            
            if i == de: 
                energy = np.arange(round(energy_i[0]+2,3),round(energy_i[-1]-2,3),0.01)   
            # JCC em 20180412: reduzimos o intervalo de energia em 1% no começo e no final para evitar erros de interpolação... Tem que melhorar...    
            
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


def XASavg(a):    #ex: pmedia = XASmedia(p)
    
    sum = 0
    
    for i in a:
        sum += i       
            
    avg = sum/(len(a))             
    
    return avg


def XASbg(yaxis,xaxis,bg): #ex: pmedia_bg = XASbg(pmedia,energy,bg)
    
    idx = SEARCHindex(xaxis,bg)
        
    bg = yaxis[idx]
    
    yaxis_bg = yaxis - bg
    
    return yaxis_bg


def XASnor(yaxis,xaxis,xas,nor):   #ex: pmedia_bg_nor = XASnor(pmedia_bg,energy,xas,nor)
                                    #ex: pmedia_bg_nor = XASnor(pmedia_bg,energy,pmedia_bg,nor)
    idx = SEARCHindex(xaxis,nor)
 
    yaxis_nor = yaxis/(xas[idx])
    
    return yaxis_nor


def CONFIGplt(ylabel='norm',xlabel='energy (eV)',grid=True):
    
    #plt.style.use('ggplot')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(grid)
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
   
    plt.plot(energy, pmedia_bg_nor, linestyle='-', color='black',linewidth=1.5, label='pos_avg'); 
    plt.plot(energy, nmedia_bg_nor, linestyle='-', color='red',linewidth=1.5, label='neg_avg'); 
    plt.plot(energy, xas_nor, linestyle='-', color='green',linewidth=0.5, label='xas_avg'); 
    plt.plot(energy, dif_nor, linestyle='-', color='blue',linewidth=1.5, label='xmcd_asymm'); 
     
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
    
    x = np.reshape(x,(r,c))
    y = np.reshape(y,(r,c))
    norm = np.reshape(norm,(r,c))
    
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
    plt.show()