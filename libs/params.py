#!/usr/bin/env python
#
# Params Library: Methods for loading, reading, writing CWITools parameter files
#
import sys

pkeys = ["NAME","RA","DEC","Z","ZLA","REG_FILE","DATA_DIR","DATA_DEPTH","PRODUCT_DIR","IMG_ID","SKY_ID","INST","XCROP","YCROP","WCROP"]      
  
#######################################################################
# Check for incomplete parameter data
def verify(params):
    global pkeys
    for pk in pkeys:
        if pk not in params:
            print(("Parameter file incomplete. Missing %s" % pk))
            sys.exit()

#######################################################################
# Parse FITS headers for relevant stacking & subtracting parameters
def parseHeaders(params,fits):

    for i,f in enumerate(fits):
        
        header = f[0].header
        
        params["INST"][i] = header["INSTRUME"] #Get instrument
        

        if params["INST"][i]=="CWI": params["INST"][i]="PCWI" #Handle old CWI name

        if params["INST"][i]=="PCWI":
            params["XCROP"][i] = "10:-12"
            params["YCROP"][i] = "0:24"
             
        elif params["INST"][i]=="KCWI":
            params["XCROP"][i] = "0:-1"
            params["YCROP"][i] = "0:-1"

        if params["INST"][i]=="PCWI":
            if header["NASMASK"]=='T' or header["NASMASK"]==True: params["SKY_ID"][i] = params["IMG_ID"][i]    
        elif params["INST"][i]=="KCWI":
            if header["BNASNAM"]=="Closed": params["SKY_ID"][i] = params["IMG_ID"][i]    
        else: 
            try:
                params["SKY_ID"][i] = header["MPIMNO"]
            except:
                params["SKY_ID"][i] = -1
          
        params["WCROP"][i] = "%i:%i" % (int(header["WAVGOOD0"]),int(header["WAVGOOD1"]+1))
        
    return params

##################################################################################################
def writeparams(params,parampath):
    global param_keys
    
    #Check for trailing '/' in directory names and add if missing
    for dirKey in ["PRODUCT_DIR","DATA_DIR"]:
        if params[dirKey][-1]!='/': params[dirKey]+='/'
    
    paramfile = open(parampath,'w')

    print(("Writing parameters to %s" % parampath))
    
   
    paramfile.write("#############################################################################################\
    \n# CWITools TARGET PARAMETER FILE \n\n""")
    paramfile.write("name         = %s   # Target name\n" % params["NAME"])
    paramfile.write("ra           = %10.8f # Decimal RA\n" % params["RA"])
    paramfile.write("dec          = %10.8f # Decimal DEC\n" % params["DEC"])
    paramfile.write("z            = %10.8f # Redshift\n" % params["Z"])
    paramfile.write("zla          = %10.8f # Lyman Alpha Redshift\n" % params["ZLA"]) 
    paramfile.write("reg_file     = %s # DS9 region file of continuum sources (default 'None')\n" % params["REG_FILE"]) 
    paramfile.write("data_dir     = %s # Location of raw data cubes\n" % params["DATA_DIR"]) 
    paramfile.write("data_depth   = %s # How many levels down from 'data_dir' to search for cubes\n" % params["DATA_DEPTH"])    
    paramfile.write("product_dir  = %s # Where to store stacked and subtracted cubes\n" % params["PRODUCT_DIR"])   
    paramfile.write("\n#############################################################################################\n")   
    paramfile.write("#%15s%15s%15s%15s%15s%15s\n" % ("IMG_ID","SKY_ID","INST","XCROP","YCROP","WCROP"))
   
    img_ids = params["IMG_ID"]
    keys = ["IMG_ID","SKY_ID","INST","XCROP","YCROP","WCROP"]
    keystr = ">%15s%15s%15s%15s%15s%15s\n"
    for key in keys:   
        if key in params and len(params[key])==len(params["IMG_ID"]): pass
        elif key=="INST": params[key] =  [ '?' for i in range(len(img_ids))]
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
    cols = []
    
    #Run through parameter file
    for line in paramfile:
        
        
        #Parse horizontal param info (key=value pairs above image table)
        if "=" in line:
        
            #Split from line
            keyval = line.split("#")[0].replace(" ","").replace("\n","")
            k,v = keyval.split("=")

            #Change key to uppercase
            k = k.upper()
            
            #Convert some values to floats
            if k in ["RA","DEC","Z","ZLA"]: v = float(v)
            
            #Add to params
            params[k] = v
        
        #Table headers - parse 
        elif "IMG_ID" in line:
        
            #Add lists to params under each column header
            cols = line.replace("#","").split()           
            for c in cols: params[c.upper()] = []
            
        #Parse table info
        elif line[0]=='>':
        
            #Split table row into values
            vals = line[1:].split()
            
            #Add to appropriate lists
            for i,v in enumerate(vals): params[cols[i]].append(v)
            
    #TEMPORARY for FLASHES data - will remove
    if "SKY_ID" not in list(params.keys()): params["SKY_ID"] = [ -1 for im in params["IMG_ID"] ]

    verify(params)
    
    for key in ["XCROP","YCROP","WCROP","INST","SKY_ID"]:
        if len(params[key])<len(params["IMG_ID"]):
            params[key] = ['-' for i in range(len(params["IMG_ID"])) ]
            
    for key in list(params.keys()):
        if key not in pkeys:
            r = input("Parameter file has outdated key values. Overwrite with new format? > ").lower()
            if r=="y" or r=="yes":
                paramfile.close()
                writeparams(params,parampath)
                return loadparams(parampath)

    #Check for trailing '/' in directory names and add if missing
    for dirKey in ["PRODUCT_DIR","DATA_DIR"]:
        if params[dirKey][-1]!='/': params[dirKey]+='/'
              
    return params
