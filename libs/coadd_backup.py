def coadd(fitsList,params,settings):

    #
    # STAGE 0: PREPARATION
    # 
    
    # Extract basic header info
    hdrList    = [ f[0].header for f in fitsList ]
    wcsList    = [ WCS(h) for h in hdrList ]
    pxScales   = np.array([ proj_plane_pixel_scales(wcs) for wcs in wcsList ])
    posAngles  = [ h["ROTPA"] for h in hdrList ]
    raQ,decQ = params["RA"],params["DEC"]

    # Get 2D headers, WCS and on-sky footprints
    h2DList    = [ get2DHeader(h) for h in hdrList]
    w2DList    = [ WCS(h) for h in h2DList ]
    footPrints = np.array([ w.calc_footprint() for w in w2DList ])

    # Exposure times
    expKeys  =  [ "TELAPSE" if inst=="KCWI" else "EXPTIME" for inst in params["INST"] ]   
    expTimes =  [ h[expKeys[i]] for i,h in enumerate(hdrList) ]

    # Extract into useful data structures
    xScales,yScales,wScales = ( pxScales[:,i] for i in range(3) )
    
    # Determine coadd scales
    coadd_xyScale = np.min(np.abs(pxScales[:,:2]))
    coadd_wScale  = np.min(np.abs(pxScales[:,2]))

    #
    # STAGE 1: WAVELENGTH ALIGNMENT
    # 
    
    #Check that the scale (Ang/px) of each input image is the same
    if len(set(wScales))!=1:
    
        print("ERROR: Wavelength axes must be equal in scale for current version of code.")
        print("Continue stacking without wavelength alignment? (y/n) >")
        answer = raw_input("")
        if not( answer=="y" or answer=="Y" or answer=="yes" ): sys.exit()
        else: print("Proceeding with stacking without any wavelength axis shifts.")
        
    else:
       
        #Get common wavelength scale
        cd33 = hdrList[0]["CD3_3"]
          
        #Get lower and upper wavelengths for each cube
        wav0s = [ h["CRVAL3"] - (h["CRPIX3"]-1)*cd33 for h in hdrList ]
        wav1s = [ wav0s[i] + h["NAXIS3"]*cd33 for i,h in enumerate(hdrList) ]
        
        #Get new wavelength axis
        wNew = np.arange(min(wav0s)-cd33, max(wav1s)+cd33,cd33)

        print("Aligning wavelength axes."),
                
        #Adjust each cube to be on new wavelenght axis
        for i,f in enumerate(fitsList):

            print('.'),
            
            #Pad the end of the cube with zeros to reach same length as wNew
            f[0].data = np.pad( f[0].data, ( (0, len(wNew)-f[0].header["NAXIS3"]), (0,0) , (0,0) ) , mode='constant' )

            #Get the wavelength offset between this cube and wNew
            dw = (wav0s[i] - wNew[0])/cd33
            
            #Split the wavelength difference into an integer and sub-pixel shift
            intShift = int(dw)
            spxShift = dw - intShift
        
            #Perform integer shift with np.roll
            f[0].data = np.roll(f[0].data,intShift,axis=0)
            
            #Create convolution matrix for subpixel shift (in effect; linear interpolation)
            K = np.array([ spxShift, 1-spxShift ])
            
            #Square interpolation coefficients for variance data
            if settings["vardata"]: K=K**2
            
            #Shift data along axis by convolving with K      
            f[0].data = np.apply_along_axis(lambda m: np.convolve(m, K, mode='same'), axis=0, arr=f[0].data)
    
            f[0].header["NAXIS3"] = len(wNew)
            f[0].header["CRVAL3"] = wNew[0]
            f[0].header["CRPIX3"] = 1
        
        print("")

         
    #   
    # Stage 2 - SPATIAL ALIGNMENT
    #
    
    # Get center and bounds of coadd canvas in RA/DEC space
    ra0,ra1   = np.max(footPrints[:,:,0]),np.min(footPrints[:,:,0])
    dec0,dec1 = np.min(footPrints[:,:,1]),np.max(footPrints[:,:,1])

    #    
    # Create header structure for coadd cube
    #
    
    # Center coordinates and pixels
    coaddHdr = hdrList[0].copy()
    coaddHdr["CRVAL1"] = ra0
    coaddHdr["CRVAL2"] = dec0
    coaddHdr["CRVAL3"] = wNew[0]        
    coaddHdr["CRPIX1"] = 1
    coaddHdr["CRPIX2"] = 1
    coaddHdr["CRPIX3"] = 1
    
    # Set position angle to zero (TO DO: Include 'mode' version of coadd code)
    coaddHdr["CD1_1"]  = -coadd_xyScale
    coaddHdr["CD2_2"]  = coadd_xyScale
    coaddHdr["CD1_2"]  = 0
    coaddHdr["CD2_1"]  = 0    
    coaddHdr["ROTPA"]  = 0
    
    # Create WCS object with this orientation & reference RA/DEC
    coaddHdr2D = get2DHeader(coaddHdr)
    coaddWCS   = WCS(coaddHdr2D)
    coaddFP    = coaddWCS.calc_footprint()
    
    # Get X,Y bounds for canvas   
    x1,y1 = coaddWCS.all_world2pix(ra1,dec1,1) 
    
    # Update Canvas size and re-generate header/WCS/footprint
    coaddHdr["NAXIS1"] = int(x1+1)
    coaddHdr["NAXIS2"] = int(y1+1)
    coaddHdr["NAXIS3"] = len(wNew)
    coaddHdr2D = get2DHeader(coaddHdr)
    coaddWCS   = WCS(coaddHdr2D)
    coaddFP    = coaddWCS.calc_footprint()

    # Create data structures to store coadded cube and corresponding exposure time mask
    coaddData = np.zeros((len(wNew),coaddHdr["NAXIS2"],coaddHdr["NAXIS1"]))
    coaddExp  = np.zeros_like(coaddData)

    #Create polygon objects once to avoid creating for each fits/pixel
    coaddPolygons = np.empty(coaddData.shape[1:],dtype=Polygon)
    for xC in range(coaddData.shape[2]):
        for yC in range(coaddData.shape[1]):
        
            #Define BL, TL, TR, BR corners of pixel as coordinates
            cPixVertices =  [ [xC-0.5,yC-0.5], [xC-0.5,yC+0.5], [xC+0.5,yC+0.5], [xC+0.5,yC-0.5] ]          
            
            #Create Polygon object and store in array  
            coaddPolygons[yC,xC] = Polygon( cPixVertices )
            
                
    # Plot footprints of each input frame and footprint of coadd frame 
    makePlots=1
    if makePlots:
        fig1,ax = plt.subplots(1,1)
        for fp in footPrints:
            ax.plot( fp[0:2,0],fp[0:2,1],'k-')
            ax.plot( fp[1:3,0],fp[1:3,1],'k-')
            ax.plot( fp[2:4,0],fp[2:4,1],'k-')
            ax.plot( [ fp[3,0], fp[0,0] ] , [ fp[3,1], fp[0,1] ],'k-')
        for fp in [coaddFP]:
            ax.plot( fp[0:2,0],fp[0:2,1],'r-')
            ax.plot( fp[1:3,0],fp[1:3,1],'r-')
            ax.plot( fp[2:4,0],fp[2:4,1],'r-')
            ax.plot( [ fp[3,0], fp[0,0] ] , [ fp[3,1], fp[0,1] ],'r-')
        ax.plot(raQ,decQ,'ro',alpha=0.8)        
        fig1.show()
        plt.waitforbuttonpress()
        plt.close()
    
        plt.ion()
        fig2,axes = plt.subplots(1,3,figsize=(18,12))
        inAx,skyAx,imgAx = axes[0:3]

                        
    # Run through each input frame
    for i,f in enumerate(fitsList):

        ##### DEBUG
        qXInput = params["SRC_X"][i] - int(params["XCROP"][i].split(':')[0]) 
        qYInput = params["SRC_Y"][i] - int(params["YCROP"][i].split(':')[0])   
        
        qRA,qDEC = w2DList[i].all_pix2world(qXInput,qYInput,0)

        qXCoadd,qYCoadd = coaddWCS.all_world2pix(qRA,qDEC,0)

        ####DEBUG

        
        #Need to handle electron counts data by converting into a 'flux' like unit
        if "electrons" in f[0].header["BUNIT"]:
            f[0].data /= expTimes[i] #Divide 'electrons' by exptime to get electrons/sec
            f[0].header["BUNIT"] = "electrons/sec" #Change units of data to a flux quantity  
        
        if makePlots:
            inAx.clear()
            skyAx.clear()
            imgAx.clear()
            inAx.set_title("Input Frame Coordinates")
            skyAx.set_title("Sky Coordinates")
            imgAx.set_title("Coadd Coordinates")
            imgAx.set_xlabel("X")
            imgAx.set_ylabel("Y")
            skyAx.set_xlabel("RA (hh.hh)")
            skyAx.set_ylabel("DEC (dd.dd)")
            inAx.plot(qXInput,qYInput,'bo',alpha=0.5)
            skyAx.plot(qRA,qDEC,'bo',alpha=0.5)
            imgAx.plot(qXCoadd,qYCoadd,'bo',alpha=0.5)
                                    
        naxis1,naxis2 = (f[0].header[k] for k in ["NAXIS1","NAXIS2"])
        wavIndices    = np.ones(f[0].data.shape[0],dtype=bool)
        wavIndices[wNew<wav0s[i]] = 0
        wavIndices[wNew>wav1s[i]] = 0
        
        #Plot footprints of just this frame and coadd frame
        if makePlots:
            xU,yU = naxis1,naxis2
            inAx.plot( [0,xU], [0,0], 'k-')
            inAx.plot( [xU,xU], [0,yU], 'k-')
            inAx.plot( [xU,0], [yU,yU], 'k-')
            inAx.plot( [0,0], [yU,0], 'k-')
            inAx.set_xlim( [-5,xU+5] )
            inAx.set_ylim( [-5,yU+5] )
            inAx.set_xlabel("X")
            inAx.set_ylabel("Y")
            
            
            xU,yU = coaddHdr["NAXIS1"],coaddHdr["NAXIS2"]
            imgAx.plot( [0,xU], [0,0], 'k-')
            imgAx.plot( [xU,xU], [0,yU], 'k-')
            imgAx.plot( [xU,0], [yU,yU], 'k-')
            imgAx.plot( [0,0], [yU,0], 'k-')
            imgAx.set_xlim( [-0.5,xU+1] )
            imgAx.set_ylim( [-0.5,yU+1] )

            #inAx.plot(qX[i],qY[i],'bo',alpha=0.6)

            #inAx.plot(xQ,yQ,'go',alpha=0.6)
            for fp in footPrints[i:i+1]:              
                skyAx.plot( fp[0:2,0],fp[0:2,1],'k-')
                skyAx.plot( fp[1:3,0],fp[1:3,1],'k-')
                skyAx.plot( fp[2:4,0],fp[2:4,1],'k-')
                skyAx.plot( [ fp[3,0], fp[0,0] ] , [ fp[3,1], fp[0,1] ],'k-')
                skyAx.plot( raQ, decQ, 'gx')

            for fp in [coaddFP]:             
                skyAx.plot( fp[0:2,0],fp[0:2,1],'r-')
                skyAx.plot( fp[1:3,0],fp[1:3,1],'r-')
                skyAx.plot( fp[2:4,0],fp[2:4,1],'r-')
                skyAx.plot( [ fp[3,0], fp[0,0] ] , [ fp[3,1], fp[0,1] ],'r-')            


              
        #Coadd-coordinates frame to build current input
        buildFrame = np.zeros_like(coaddData)
        
        #Parallel frame storing 'fraction' coefficients (think of better explanation)
        fractFrame = np.zeros_like(coaddData)
                
        print("Mapping %s to coadd frame (%i/%i)"%(params["IMG_ID"][i],i+1,len(fitsList))),
        
        #Loop through spatial pixels in this input frame
        for yi0 in range(f[0].data.shape[1]):
            print("."),
            sys.stdout.flush()
            for xi0 in range(f[0].data.shape[2]):
               
                yi1,xi1 = yi0+1,xi0+1
                
                #Get four vertices of this pixel (xPixel_Input)
                #Defining these vertices in a clockwise or counter-clockwise pattern (not zig-zag) is important!
                xPV_In = np.array([ xi1-0.5, xi1+0.5, xi1+0.5, xi1-0.5 ])
                yPV_In = np.array([ yi1-0.5, yi1-0.5, yi1+0.5, yi1+0.5 ])
                
                #Convert these vertices to RA/DEC positions
                ras,decs = w2DList[i].all_pix2world(xPV_In,yPV_In,1)
                
                #Now convert these vertices to image coordinates in the coadd frame
                xPV_Coadd,yPV_Coadd = coaddWCS.all_world2pix(ras,decs,1)
                
                if makePlots:
                    skyAx.plot(ras,decs,'kx')
                    imgAx.plot(xPV_Coadd-1,yPV_Coadd-1,'kx')
     
                #Create polygon object for projection of this input pixel onto coadd grid
                pixIN = Polygon( [ [ xPV_Coadd[j], yPV_Coadd[j] ] for j in range(len(xPV_Coadd)) ] )                 
                
                #Get bounding pixels on coadd grid  
                xP0,yP0,xP1,yP1 = (int(x) for x in list(pixIN.bounds))      

                #Run through pixels on coadd grid and add input data
                for xC in range(xP0,xP1+1):
                    for yC in range(yP0,yP1+1):

                        #Get polygon for this coadd frame pixel
                        pixCA = coaddPolygons[yC-1,xC-1]

                        #Calculation fractional overlap between input/coadd pixels
                        overlap = pixIN.intersection(pixCA).area/pixIN.area
                       
                        #Add fraction to fraction frame
                        fractFrame[wavIndices,yC-1,xC-1] += overlap

                        #Square coefficient if variance data is being stacked
                        if settings["vardata"]: overlap=overlap**2
                        
                        #Add data to build frame
                        buildFrame[:,yC-1,xC-1] += overlap*f[0].data[:,yi0,xi0]

        #Get mask of non-zero voxels in build frame
        M = fractFrame<1
        
        #Get the ratio of coadd pixel size to input pixel size
        f0 = round((coadd_xyScale**2)/(xScales[i]*yScales[i]),4)

        #Trim edge pixels (and also change all 0s to 1s to avoid NaNs)
        ff = fractFrame.flatten()
        bb = buildFrame.flatten()
        bb[ff<f0] = 0
        ff[ff<f0] = 1
        fractFrame = np.reshape(ff,coaddData.shape)
        buildFrame = np.reshape(bb,coaddData.shape)
               
        #Create 3D mask of observations
        M = np.reshape( ff<1, coaddData.shape)

        #Add weight*data to coadd (numerator of weighted mean with exptime as weight)
        if settings["vardata"]: coaddData += (expTimes[i]**2)*buildFrame
        else: coaddData += expTimes[i]*buildFrame
        
        #Add to exposure mask
        coaddExp += expTimes[i]*M

        #Add weights to mask (denominator of weighted mean)     
        if makePlots:
            fig2.canvas.draw()
            raw_input("")
            #plt.waitforbuttonpress()
        
        print("")
        
    if makePlots: plt.close()
    
    #Convert 0s to 1s in exposure time cube
    ee = coaddExp.flatten()
    ee[ee==0] = 1
    coaddExp = np.reshape( ee, coaddData.shape )
    
    #Divide by sum of weights (or square of sum)
    if settings["vardata"]: coaddData /= coaddExp**2
    else:  coaddData /= coaddExp
    
    #Create FITS object
    coaddHDU = apIO.fits.PrimaryHDU(coaddData)
    coaddFITS = apIO.fits.HDUList([coaddHDU])
    coaddFITS[0].header = coaddHdr

    return coaddFITS
