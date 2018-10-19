#!/usr/bin/env python
#
# Params Library: Methods for loading, reading, writing CWITools parameter files
#

#######################################################################  
#Return indices for a given wavelength band (w1,w2) in angstroms
def getband(_w1,_w2,_hd):
	w0,dw,p0 = _hd["CRVAL3"],_hd["CD3_3"],_hd["CRPIX3"]
	w0 -= p0*dw
	return ( int((_w1-w0)/dw), int((_w2-w0)/dw) )
    
#######################################################################
# Check for incomplete parameter data
def paramsMissing(params):
    #Search for default values in params dictionary and return 'incomplete' flag if found
    if '?' in params["INST"]  or\
        -1 in params["SRC_X"] or\
        -1 in params["SRC_Y"] or\
       '-' in params["WCROP"] or\
       '-' in params["XCROP"] or\
       '-' in params["YCROP"] or\
        -1 in params["PA"]:
        return True
    else: return False
    
    
#######################################################################
# Parse FITS headers for relevant stacking & subtracting parameters
def parseHeaders(params,fits):
    
    wg0,wg1 = [],[]
    for i,f in enumerate(fits):
        
        header = f[0].header
        
        params["INST"][i] = header["INSTRUME"] #Get instrument
        
        if params["INST"][i]=="CWI": params["INST"][i]="PCWI" #Handle old CWI name
               
        #Hardcode these for now - update later to use inst.config files
        if params["INST"][i]=="PCWI":
            params["PA"][i] = int(header["ROTPA"])
            params["XCROP"][i] = "10:-12"
            params["YCROP"][i] = "0:24"
             
        elif params["INST"][i]=="KCWI":
            params["PA"][i] = int(header["ROTPOSN"])
            params["XCROP"][i] = "0:-1"
            params["YCROP"][i] = "0:-1"

        if header["NASMASK"]=='T': params["SKY"][i] = params["IMG_ID"][i]
        else: 
            try:
                params["SKY_ID"][i] = header["MPIMNO"]
            except:
                params["SKY_ID"][i] = -1                
        wg0.append(header["WAVGOOD0"])
        wg1.append(header["WAVGOOD1"])
        
    #Get maximum overlapping good wavelength range of all input cubes
    wg0max = max(wg0)
    wg1min = min(wg1)       
    
    #Run through cubes again and crop to this wavelength range
    for i,f in enumerate(fits): params["WCROP"][i] = "%i:%i" % getband(wg0max,wg1min,f[0].header)

    return params

##################################################################################################
def writeparams(params,parampath):
    global param_keys
    
    paramfile = open(parampath,'w')

    print("Writing parameters to %s" % parampath)
      
    paramfile.write("#############################################################################################\
    \n# TARGET PARAMETER FILE FOR STACKING & SUBTRACTING (KiLLER Pipeline)\n\n""")

    paramfile.write("name = %s # Target name\n" % params["NAME"])
    paramfile.write("ra   = %.6f # Decimal RA\n" % params["RA"])
    paramfile.write("dec  = %.6f # Decimal DEC\n" % params["DEC"])
    paramfile.write("z    = %.6f # Redshift\n" % params["Z"])
    paramfile.write("zla  = %.6f # Lyman Alpha Redshift\n\n" % params["ZLA"])
    
    paramfile.write("reg_file = %s # DS9 region file of continuum sources (default 'None')\n\n" % params["REG_FILE"])
    
    paramfile.write("data_dir = %s # Location of raw data cubes\n\n" % params["DATA_DIR"])
    
    paramfile.write("data_depth = %s # How many levels down from 'data_dir' to search for cubes\n\n" % params["DATA_DEPTH"])
    
    paramfile.write("product_dir = %s # Where to store stacked and subtracted cubes\n\n" % params["PRODUCT_DIR"])
    
    paramfile.write("#############################################################################################\n")   
    paramfile.write("#%15s%10s%10s%10s%10s%10s%15s%15s%15s\n"\
    % ("IMG_ID","SKY_ID","INST","PA","SRC_X","SRC_Y","XCROP","YCROP","WCROP"))
    
    img_ids = params["IMG_ID"]
    keys = ["IMG_ID","SKY_ID","INST","PA","SRC_X","SRC_Y","XCROP","YCROP","WCROP"]
    keystr = ">%15s%10s%10s%10i%10.2f%10.2f%15s%15s%15s\n"
    for key in keys:
    
        if params.has_key(key) and len(params[key])==len(params["IMG_ID"]): pass
        elif key=="INST": params[key] =  [ '?' for i in range(len(img_ids))]
        elif key=="PA": params[key] =  [ -1 for i in range(len(img_ids))]
        elif "CROP" in key: params[key] = [ '0:-1' for i in range(len(img_ids))]
        elif key=="SKY_ID": params[key] = [ '-1' for i in range(len(img_ids)) ]
        else: params[key] = [ -99 for i in range(len(img_ids))] 

    for i in range(len(img_ids)): paramfile.write( keystr % tuple(params[keys[j]][i] for j in range(len(keys))))
    paramfile.close()
    
    
##################################################################################################   
def loadparams(parampath):
    
    paramfile = open(parampath,'r')

    #print("Loading target parameters from %s" % parampath)
    
    params = {}
    params["IMG_ID"] = []
    params["SKY_ID"] = []
    params["INST"]   = []
    params["PA"]     = []
    params["SRC_X"]  = []
    params["SRC_Y"]  = []
    params["XCROP"]  = []
    params["YCROP"]  = []
    params["WCROP"]  = []

    for line in paramfile:
    
        if "=" in line:
            keyval = line.split("#")[0].replace(" ","").replace("\n","")
            key,val = keyval.split("=")
            params[key.upper()] = float(val) if key in ['ra','dec','z','zla'] else val

        elif line[0]=='>' and len(line[1:].split())>=8:

            img_id,sky_id,inst,pa,qsox,qsoy,xcrop,ycrop,wcrop = line[1:].split()[0:9]
            
            params["IMG_ID"].append(img_id)
            
            params["SKY_ID"].append(sky_id)
            
            params["INST"].append(inst)
            
            params["PA"].append(int(pa))
            
            params["SRC_X"].append(float(qsox))
            
            params["SRC_Y"].append(float(qsoy))
            
            params["XCROP"].append(xcrop)

            params["YCROP"].append(ycrop)
                                  
            params["WCROP"].append(wcrop)
            
        elif line[0]=='>' and len(line.split())==2:
        
            params["IMG_ID"].append(line[1:].split()[0])

    #If no region file specified yet
    if not params.has_key("REG_FILE"): params["REG_FILE"] = "None"
    
    #If some image numbers have been added but not properly written to param file...
    if len(params["IMG_ID"]) > len(params["SRC_Y"]):
    
        paramfile.close() #Close the parameter file
        
        writeparams(params,parampath) #Rewrite the file to fit the correct format
    
        return loadparams(parampath) #Try to reload params and return those
        
    #Otherwise just return the loaded params
    else: return params
