"""Generic tools for saving files, etc."""
from astropy.io import fits
from astropy import units as u
from astropy import wcs
import cwitools
import numpy as np
from scipy import ndimage
import os
import pkg_resources
import sys
import warnings
from cwitools import coordinates
from PyAstronomy import pyasl
from reproject import reproject_interp
from reproject import reproject_exact
import matplotlib.pyplot as plt


clist_template = {
    "INPUT_DIRECTORY":"./",
    "SEARCH_DEPTH":3,
    "OUTPUT_DIRECTORY":"./",
    "ID_LIST":[]
}

def get_instrument(hdr):
    if 'INSTRUME' in hdr:
        return hdr['INSTRUME']
    else:
        raise ValueError("Instrument not recognized.")

def get_specres(hdr):

    inst = get_instrument(hdr)

    if inst == 'PCWI':
        if 'MEDREZ' in hdr['GRATID']: return 2500
        else: return 5000

    elif inst == 'KCWI':

        grating, slicer = hdr['BGRATNAM'], hdr['IFUNAM']

        if grating == 'BL':
            R0 = 900
        elif grating == 'BM':
            R0 = 2000
        elif 'BH' in grating:
            R0 = 4500
        else:
            raise ValueError("Grating not recognized (header:BGRATNAM)")

        if slicer == 'Small':
            mul = 4
        elif slicer == 'Medium':
            mul = 2
        elif slicer == 'Large':
            mul = 1
        else:
            raise ValueError("Slicer not recognized (header:IFUNAM)")

        return mul * R0

    else:
        raise ValueError("Instrument not recognized.")

def get_skylines(inst, use_vacuum=False):

    if inst == 'PCWI':
        sky_file = 'palomar_lines.txt'
    elif inst == 'KCWI':
        sky_file = 'keck_lines.txt'
    else:
        raise ValueError("Instrument not recognized.")

    data_path = pkg_resources.resource_stream(__name__, 'data/sky/%s'% sky_file)
    data = np.loadtxt(data_path)
    
    if use_vacuum:
        data = pyasl.airtovac2(data)

    return data

def get_skymask(hdr):
    """Get mask of sky lines for specific instrument/resolution."""
    wav_type=hdr['CTYPE3']
    if wav_type=='AWAV':
        use_vacuum=False
    elif wav_type=='WAVE':
        use_vacuum=True
    else:
        raise ValueError("Wave type not recognized.")
    
    wav_axis = coordinates.get_wav_axis(hdr)
    wav_mask = np.zeros_like(wav_axis, dtype=bool)
    inst = get_instrument(hdr)
    res = get_specres(hdr)
    skylines = get_skylines(inst, use_vacuum=use_vacuum)

    for line in skylines:
        dlam = 1.4 * line / res #Get width of line from inst res.
        wav_mask[np.abs(wav_axis - line) <= dlam] = 1
    return wav_mask

def get_skybins(hdr):
    """Get sky-line masks in 2D bins."""
    wav_type=hdr['CTYPE3']
    if wav_type=='AWAV':
        use_vacuum=False
    elif wav_type=='WAVE':
        use_vacuum=True
    else:
        raise ValueError("Wave type not recognized.")
    inst = get_instrument(hdr)
    res = get_specres(hdr)
    skylines = get_skylines(inst, use_vacuum=use_vacuum)
    bin_list = []
    for line in skylines:
        onebin = [line-1.4*line/res, line+1.4*line/res]
        bin_list.append(onebin)
    return bin_list

def bunit_todict(st):
    """Convert BUNIT string to a dictionary"""
    numchar=[str(i) for i in range(10)]
    numchar.append('+')
    numchar.append('-')
    dictout={}
    
    st_list=st.split()
    for st_element in st_list:
        flag=0
        for i,char in enumerate(st_element):
            if char in numchar:
                flag=1
                break
        
        if i==0:
            key=st_element
            power_st='1'
        elif flag==0:
            key=st_element
            power_st='1'
        else:
            key=st_element[0:i]
            power_st=st_element[i:]
        
        dictout[key]=float(power_st)
    
    return dictout

def get_bunit(hdr):
    """"Get BUNIT string that meets FITS standard."""
    bunit=multiply_bunit(hdr['BUNIT'])
    
    return bunit
    
def multiply_bunit(bunit,multiplier='1'):
    """Unit conversions and multiplications."""
    
    # electrons
    electron_power=0.
    if 'electrons' in bunit:
        bunit=bunit.replace('electrons','1')
        electron_power=1
    if 'variance' in bunit:
        bunit=bunit.replace('variance','1')
        electron_power=2
    
    # Angstrom
    if '/A' in bunit:
        bunit=bunit.replace('/A','/angstrom')

    # unconventional expressions
    if 'FLAM' in bunit:
        addpower=1
        if '**2' in bunit:
            addpower=2
            bunit=bunit.replace('**2','')
        power=float(bunit.replace('FLAM',''))
        v0=u.erg/u.s/u.cm**2/u.angstrom*10**(-power)            
        v0=v0**addpower
    elif 'SB' in bunit:
        addpower=1
        if '**2' in bunit:
            addpower=2
            bunit=bunit.replace('**2','')
        power=float(bunit.replace('SB',''))
        v0=u.erg/u.s/u.cm**2/u.angstrom/u.arcsec**2*10**(-order)
        v0=v0**addpower
    else:
        v0=u.Unit(bunit)

    if type(multiplier)==type(''):
        if 'A' in multiplier:
            multiplier=multiplier.replace('A','angstrom')
        multi=u.Unit(multiplier)
    else:
        multi=multiplier
                
    vout=(v0*multi)
    # convert to quantity
    if type(vout)==type(u.Unit('erg/s')):
        vout=u.Quantity(1,vout)
    vout=vout.cgs
    stout="{0.value:.0e} {0.unit:FITS}".format(vout)
    stout=stout.replace('1e+00 ','')
    stout=stout.replace('10**','1e')
    dictout=bunit_todict(stout)
    
    # clean up
    if 'rad' in dictout:
        vout=(vout*u.arcsec**(-dictout['rad'])).cgs*u.arcsec**dictout['rad']
        stout="{0.value:.0e} {0.unit:FITS}".format(vout)
        dictout=bunit_todict(stout)
    
    if 'Ba' in dictout:
        vout=vout*(u.Ba**(-dictout['Ba']))*(u.erg/u.cm**3)**dictout['Ba']
        stout="{0.value:.0e} {0.unit:FITS}".format(vout)
        dictout=bunit_todict(stout)

    if 'g' in dictout:
        vout=vout*(u.g**(-dictout['g']))*(u.erg*u.s**2/u.cm**2)**dictout['g']
        stout="{0.value:.0e} {0.unit:FITS}".format(vout)
        dictout=bunit_todict(stout)
    
    # electrons
    if electron_power>0:
        stout=stout+' electrons'+'{0:.0f}'.format(electron_power)+' '
        dictout=bunit_todict(stout)
    
    # sort
    def unit_key(st):
        if st[0] in [str(i) for i in np.arange(10)]:
            return 0
        elif 'erg' in st:
            return 1
        elif 'electrons' in st:
            return 1
        elif st[0]=='s':
            return 2
        elif 'cm' in st:
            return 3
        elif 'arcsec' in st:
            return 4
        else:
            return 5
    st_list=stout.split()
    st_list.sort(key=unit_key)
    stout=' '.join(st_list)
    
    return stout

def extractHDU(fits_in):
    type_in = type(fits_in)
    if type_in == fits.HDUList:
        return fits_in[0]
    elif type_in == fits.ImageHDU or type_in == fits.PrimaryHDU:
        return fits_in
    else:
        raise ValueError("Astropy ImageHDU, PrimaryHDU or HDUList expected.")

def matchHDUType(fits_in, data, header):
    """Return a HDU or HDUList with data/header matching the type of the input."""
    type_in = type(fits_in)
    if type_in == fits.HDUList:
        return fits.HDUList([fits.PrimaryHDU(data, header)])
    elif type_in == fits.ImageHDU:
        return fits.ImageHDU(data, header)
    elif type_in == fits.PrimaryHDU:
        return fits.PrimaryHDU(data, header)
    else:
        raise ValueError("Astropy ImageHDU, PrimaryHDU or HDUList expected.")

def get_fits(data, header=None):
    hdu = fits.PrimaryHDU(data, header=header)
    hdulist = fits.HDUList([hdu])
    return hdulist

def set_cmdlog(path):
    cwitools.command_log = path


def find_files(id_list, datadir, cubetype, depth=3):
    """Finds the input files given a CWITools parameter file and cube type.

    Args:
        params (dict): CWITools parameters dictionary.
        cubetype (str): Type of cube (e.g. icubes.fits) to load.

    Returns:
        list(string): List of file paths of input cubes.

    Raises:
        NotADirectoryError: If the input directory does not exist.

    """

    #Check data directory exists
    if not os.path.isdir(datadir):
        raise NotADirectoryError("Data directory (%s) does not exist. Please correct and try again." % datadir)

    #Load target cubes
    N_files = len(id_list)
    target_files = []
    typeLen = len(cubetype)

    for root, dirs, files in os.walk(datadir):

        if root[-1] != '/': root += '/'
        rec = root.replace(datadir, '').count("/")

        if rec > depth: continue
        else:
            for f in files:
                if f[-typeLen:] == cubetype:
                    for i,ID in enumerate(id_list):
                        if ID in f:
                            target_files.append(root + f)

    #Print file paths or file not found errors
    if len(target_files) < len(id_list):
        warnings.warn("Some files were not found:")
        for id in id_list:
            is_in = np.array([ id in x for x in target_files])
            if not np.any(is_in):
                warnings.warn("Image with ID %s and type %s not found." % (id, cubetype))


    return sorted(target_files)

def parse_cubelist(filepath):
    """Load a CWITools parameter file into a dictionary structure.

    Args:
        path (str): Path to CWITools .list file

    Returns:
        dict: Python dictionary containing the relevant fields and information.

    """
    global clist_template
    clist = {k:v for k, v in clist_template.items()}

    #Parse file
    listfile = open(filepath, 'r')
    for line in listfile:

        line = line[:-1] #Trim new-line character
        #Skip empty lines
        if line == "":
            continue

        #Add IDs when indicated by >
        elif line[0] == '>':
            clist["ID_LIST"].append(line.replace('>', ''))

        elif '=' in line:

            line = line.replace(' ', '')     #Remove white spaces
            line = line.replace('\n', '')    #Remove line ending
            line = line.split('#')[0]        #Remove any comments
            key, val = line.split('=') #Split into key, value pair
            if key.upper() in clist:
                clist[key] = val
            else:
                raise ValuError("Unrecognized cube list field: %s" % key)
    listfile.close()

    #Perform quick validation of input, but only warn for issues
    input_isdir = os.path.isdir(clist["INPUT_DIRECTORY"])
    if not input_isdir:
        warnings.warn("%s is not a directory." % clist["INPUT_DIRECTORY"])

    output_isdir = os.path.isdir(clist["OUTPUT_DIRECTORY"])
    if not output_isdir:
        warnings.warn("%s is not a directory." % clist["OUTPUT_DIRECTORY"])

    try:
        clist["SEARCH_DEPTH"] = int(clist["SEARCH_DEPTH"])
    except:
        raise ValuError("Could not parse SEARCH_DEPTH to int (%s)" % clist["SEARCH_DEPTH"])
    #Return the dictionary
    return clist

def output(str, log=None, silent=None):

    uselog = True

    #First priority, take given log
    if log != None:
        logfilename = log

    #Second priority, take global log file
    elif cwitools.log_file != None:
        logfilename = cwitools.log_file

    #If neither log set, ignore
    else:
        uselog = False

    #If silent is actively set to False by function call
    if silent == False:
        print(str, end='')

    #If silent is not set, but global 'silent_mode' is False
    elif silent == None and cwitools.silent_mode == False:
        print(str, end='')

    else: pass

    if uselog:
        logfile = open(logfilename, 'a')
        logfile.write(str)
        logfile.close()


def xcor_2d(hdu0, hdu1, preshift=[0,0], maxstep=None, box=None, upscale=1, conv_filter=2.,
            background_subtraction=False, background_level=None, reset_center=False,
            method='interp-bicubic', output_flag=False, plot=0):
    """Perform 2D cross correlation to image HDUs and returns the relative shifts."""
    
    if 'interp' in method:
        _,interp_method=method.split('-')
        def tmpfunc(hdu1,header):
            return reproject_interp(hdu1,header,order=interp_method)
        reproject_func=tmpfunc
    elif 'exact' in method:
        reproject_func=reproject_exact
    else:
        raise ValueError('Interpolation method not recognized.')
        
    upscale=int(upscale)
    
    # Properties
    hdu1_old=hdu1
    hdu0=hdu0.copy()
    hdu1=hdu1.copy()
    sz0=hdu0.shape
    sz1=hdu1.shape
    wcs0=wcs.WCS(hdu0.header)
    wcs1=wcs.WCS(hdu1.header)
    
    old_crpix1=[hdu1.header['CRPIX1'],hdu1.header['CRPIX2']]

    # defaults
    if maxstep is None:
        maxstep=[sz1[1]/4.,sz1[0]/4.]
    maxstep=[int(np.round(i)) for i in maxstep]
        
    if box is None:
        box=[0,0,sz0[1],sz0[0]]
        
    if reset_center:
        ad_center0=wcs0.all_pix2world(sz0[1]/2+0.5,sz0[0]/2+0.5,0)
        ad_center0=[float(i) for i in ad_center0]
        
        xy_center0to1=wcs1.all_world2pix(*ad_center0,0)
        xy_center0to1=[float(i) for i in xy_center0to1]
        
        dcenter=[(sz1[1]/2+0.5)-xy_center0to1[0],(sz1[0]/2+0.5)-xy_center0to1[1]]
        hdu1.header['CRPIX1']+=dcenter[0]
        hdu1.header['CRPIX2']+=dcenter[1]
        wcs1=wcs.WCS(hdu1.header)
    
    # preshifts
    hdu1.header['CRPIX1']+=preshift[0]
    hdu1.header['CRPIX2']+=preshift[1]
    
    # upscale
    def hdu_upscale(hdu,upscale,header_only=False):
        hdu_up=hdu.copy()
        if upscale!=1:
            hdr_up=hdu_up.header
            hdr_up['NAXIS1']=hdr_up['NAXIS1']*upscale
            hdr_up['NAXIS2']=hdr_up['NAXIS2']*upscale
            hdr_up['CRPIX1']=(hdr_up['CRPIX1']-0.5)*upscale+0.5
            hdr_up['CRPIX2']=(hdr_up['CRPIX2']-0.5)*upscale+0.5
            hdr_up['CD1_1']=hdr_up['CD1_1']/upscale
            hdr_up['CD2_1']=hdr_up['CD2_1']/upscale
            hdr_up['CD1_2']=hdr_up['CD1_2']/upscale
            hdr_up['CD2_2']=hdr_up['CD2_2']/upscale
            if not header_only:
                hdu_up.data,coverage=reproject_func(hdu,hdr_up)

        return hdu_up
    
    hdu0=hdu_upscale(hdu0,upscale)
    hdu1=hdu_upscale(hdu1,upscale)
    
    
    # project 1 to 0
    img1,cov1=reproject_func(hdu1,hdu0.header)
    
    
    img0=np.nan_to_num(hdu0.data,nan=0,posinf=0,neginf=0)
    img1=np.nan_to_num(img1,nan=0,posinf=0,neginf=0)
    img1_expand=np.zeros((sz0[0]*3*upscale,sz0[1]*3*upscale))
    img1_expand[sz0[0]*upscale:sz0[0]*2*upscale,sz0[1]*upscale:sz0[1]*2*upscale]=img1
    
    # +/- maxstep pix
    xcor_size=((np.array(maxstep)-1)*upscale+1)+int(np.ceil(conv_filter))
    xx=np.linspace(-xcor_size[0],xcor_size[0],2*xcor_size[0]+1,dtype=int)
    yy=np.linspace(-xcor_size[1],xcor_size[1],2*xcor_size[1]+1,dtype=int)
    dy,dx=np.meshgrid(yy,xx)
    
    xcor=np.zeros(dx.shape)
    for ii in range(xcor.shape[0]):
        for jj in range(xcor.shape[1]):
            cut0=img0[box[1]*upscale:box[3]*upscale,box[0]*upscale:box[2]*upscale]
            cut1=img1_expand[box[1]*upscale-dy[ii,jj]+sz0[0]*upscale:box[3]*upscale-dy[ii,jj]+sz0[0]*upscale,
                             box[0]*upscale-dx[ii,jj]+sz0[1]*upscale:box[2]*upscale-dx[ii,jj]+sz0[1]*upscale]
            if background_subtraction:
                if background_level is None:
                    back_val0=np.median(cut0[cut0!=0])
                    back_val1=np.median(cut1[cut1!=0])
                else:
                    back_val0=float(background_level[0])
                    back_val1=float(background_level[1])
                cut0=cut0-back_val0
                cut1=cut1-back_val1
            else:
                if not background_level is None:
                    cut0[cut0<background_level[0]]=0
                    cut1[cut1<background_level[1]]=0
            
            cut0[cut0<0]=0
            cut1[cut1<0]=0
            mult=cut0*cut1
            if np.sum(mult!=0)>0:
                xcor[ii,jj]=np.sum(mult)/np.sum(mult!=0)     
                
    # local maxima
    max_conv=ndimage.filters.maximum_filter(xcor,2*conv_filter+1)
    maxima=(xcor==max_conv)
    labeled, num_objects=ndimage.label(maxima)
    slices=ndimage.find_objects(labeled)
    xindex,yindex=[],[]
    for dx,dy in slices:
        x_center=(dx.start+dx.stop-1)/2
        xindex.append(x_center)
        y_center=(dy.start+dy.stop-1)/2
        yindex.append(y_center)
    xindex=np.array(xindex).astype(int)
    yindex=np.array(yindex).astype(int)
    # remove boundary effect
    index=((xindex>=conv_filter) & (xindex<2*xcor_size[0]-conv_filter) & 
            (yindex>=conv_filter) & (yindex<2*xcor_size[1]-conv_filter))
    xindex=xindex[index]
    yindex=yindex[index]
    # closest one
    if len(xindex)==0:
        # Error handling
        if output_flag==True:
            return 0.,0.,False
        else:
            # perhaps we can use the global maximum here, but it is also garbage...
            raise ValueError('Unable to find local maximum in the XCOR map.')
        
    max=np.max(max_conv[xindex,yindex])
    med=np.median(xcor)
    index=np.where(max_conv[xindex,yindex] > 0.3*(max-med)+med)
    xindex=xindex[index]
    yindex=yindex[index]
    r=(xx[xindex]**2+yy[yindex]**2)
    index=r.argmin()
    xshift=xx[xindex[index]]/upscale
    yshift=yy[yindex[index]]/upscale
    
    hdu1=hdu_upscale(hdu1,1/upscale,header_only=True)
    x_final=hdu1.header['CRPIX1']+xshift-old_crpix1[0]
    y_final=hdu1.header['CRPIX2']+yshift-old_crpix1[1]
    
    plot=int(plot)
    if plot!=0:
        if plot==1:
            fig,axes=plt.subplots(figsize=(6,6))
        elif plot==2:
            fig,axes=plt.subplots(3,2,figsize=(8,12))
        else:
            raise ValueError('Allowed values for "plot": 0, 1, 2.')
        
        # xcor map
        if plot==2:
            ax=axes[0,0]
        elif plot==1:
            ax=axes
        xplot=(np.append(xx,xx[1]-xx[0]+xx[-1])-0.5)/upscale
        yplot=(np.append(yy,yy[1]-yy[0]+yy[-1])-0.5)/upscale
        colormesh=ax.pcolormesh(xplot,yplot,xcor.T)
        xlim=ax.get_xlim()
        ylim=ax.get_ylim()
        ax.plot([xplot.min(),xplot.max()],[0,0],'w--')
        ax.plot([0,0],[yplot.min(),yplot.max()],'w--')
        ax.plot(xshift,yshift,'+',color='r',markersize=20)
        ax.set_xlabel('dx')
        ax.set_ylabel('dy')
        ax.set_title('XCOR_MAP')
        fig.colorbar(colormesh,ax=ax)
        
        if plot==2:
            fig.delaxes(axes[0,1])

            # adu0
            cut0_plot=img0[box[1]*upscale:box[3]*upscale,box[0]*upscale:box[2]*upscale]
            ax=axes[1,0]
            imshow=ax.imshow(cut0_plot,origin='bottom')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Ref img')
            fig.colorbar(imshow,ax=ax)

            # adu1
            cut1_plot=img1_expand[box[1]*upscale+sz0[0]*upscale:box[3]*upscale+sz0[0]*upscale,
                                 box[0]*upscale+sz0[1]*upscale:box[2]*upscale+sz0[1]*upscale]
            ax=axes[1,1]
            imshow=ax.imshow(cut1_plot,origin='bottom')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Original img')
            fig.colorbar(imshow,ax=ax)


            # sub1
            ax=axes[2,0]
            imshow=ax.imshow(cut1_plot-cut0_plot,origin='bottom')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Original sub')
            fig.colorbar(imshow,ax=ax)

            
            # sub2
            cut1_best=img1_expand[(box[1]+sz0[0]-int(yshift))*upscale:(box[3]+sz0[0]-int(yshift))*upscale,
                                  (box[0]+sz0[1]-int(xshift))*upscale:(box[2]+sz0[1]-int(xshift))*upscale]
            ax=axes[2,1]
            imshow=ax.imshow(cut1_best-cut0_plot,origin='bottom')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Best sub')
            fig.colorbar(imshow,ax=ax)

    
        fig.tight_layout()
        plt.show()
        
    if output_flag==True:
        return x_final,yfinal,True
    else:
        return x_final,y_final

def diagnosticPcolor(data):
    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax  = plt.subplots(1, 1)
    ax.pcolor(data)
    #ax.contour(data)
    fig.show()
    plt.waitforbuttonpress()
    plt.close()
