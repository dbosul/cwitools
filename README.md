# KiLLER Pipeline

The Kinematically-linked Line Emission Regions (KiLLER) Pipeline is a data anlysis pipeline created with the specific goal of extracting and analysing faint, extended line emission from 3D IFU data cubes. 

It is written in Python (version 2.7 at the moment - being updated to Python 3) for ease of access and open-source collaboration down the line.

At the moment, I have compiled some of the core functions of the pipeline in a compact version called quickTools, which enables a few simple command-line operations. The full version of the pipeline will also act as a python library/package to give more flexibility to the user.

# Using quickTools

There are two scripts to note in quickTools: killer_stack.py and killer_subtract.py

These are both used on the command line with the syntax:

"python <script_name> <parameter_file> <cube_type>"

<script_name> would obviously be killer_stack.py or killer_subtract.py

<parameter_file> is a file containing all the relevant information to construct a stacked cube. An example of the contents of a parameter file is provided in the quickTools/ folder.

<cube_type> is an identifier string to point the pipeline towards which type of data cube you want to stack. For example, if working with PCWI or KCWI data, you may want to stack the icube, icuber, icubes or any other custom variant you've modified yourself. It helps to include the file extension '.fits' in this parameter to avoid confusion with any other files. 


A few things should be pointed out in general:

1. The pipeline gets parameters and alignment for stacking using a continuum source such as a QSO. This means whatever you stack first after creating a new parameter file must contain a bright enough continuum source. 

2. QSO subtraction happens at the individual cube level, and subtracted cubes are saved with the same filenames as the input cubes with "_qsub.fits" appended. You then need to stack these cubes separately.

3. If any alignment or geometry parameters are missing from the parameter file - the pipeline will re-run the alignment process.

# Creating a new parameter file

To create a new parameter file - just make a copy of the template.param file provided and modify the following values:

Target Name - Will be used as the name of the stacked cube. 

RA, DEC - Both decimal degrees. Used to correct WCS.

z - Redshift. Currently used to eliminate wavelengths around Lyman-alpha +/- 2000km/s for continuum fitting procedures.

data_dir - The top-level directory where the pipeline should look for the reduced data cubes

data_depth  - The number of levels down from data_dir the pipeline should search. 

E.g. if your data is stored in a directory structure like "/home/user/data/kcwi/<date>/redux/" then you would set data_dir="/home/user/data/kcwi/" and data_depth=2 to reach the 'redux' directories in each date directory. Or if all the data you need is already in a designated target folder like "/home/user/data/UM287/" then you would set data_dir to that directory and data_depth=0.

prod_dir - The 'product' directory tells the pipeline where to save the final stacked cubes.

Once this information is provided - all you need to do is add a unique identifier string for each image you want to stack. In PCWI data, this is just the image number (e.g. "18792") while in KCWI data this would be a <date_imgNumber> string like "171018_00123"

Add one of these per line, below the row of column headers ("IMG_ID   INST    PA .... " etc) with the character '>' at the start of each row.

E.g. to add PCWI images 18792, 18974, 18796

>   18792
>   18794
>   18796

or for KCWI images 77-79 from the date 170916

>   170916_00077
>   170916_00078
>   170916_00079

You don't need to add anything else to these lines. The pipeline will fill in the values automatically once you run the stacking procedure.



