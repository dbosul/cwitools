"""Initialize a CWITools parameter file for a new target."""
from cwitools import parameters

import argparse
import warnings

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Measure WCS parameters and save to WCS correction file.')
    parser.add_argument('-targetname',
                        type=str,
                        help="Target name"
    )
    parser.add_argument('-ra',
                        type=float,
                        help="Target right-ascension (degrees)."
    )
    parser.add_argument('-dec',
                        type=float,
                        help="Target declination (degrees)."
    )
    parser.add_argument('-input_dir',
                        type=str,
                        help="Top-level directory containing input data."
    )
    parser.add_argument('-search_depth',
                        type=int,
                        help="Recursive search depth to use when looking for files in -input_dir"
    )
    parser.add_argument('-output_dir',
                        type=float,
                        help="Directory to save output files in."
    )
    parser.add_argument('-id_list',
                        type=float,
                        help="Comma separated list of unique identifiers for input files (e.g. image12345 or kb190303_00011)"
    )
    parser.add_argument('-out',
                        type=str,
                        help="Output file name."
    )
    args = parser.parse_args()

    #Initialize a parameters dictionary
    params = parameters.init_params()
    p_types = parameters.parameter_fields

    #Get associations between input arguments and param fields
    input_dict ={
        "TARGET_NAME": args.targetname,
        "TARGET_RA": args.ra,
        "TARGET_DEC": args.dec,
        "INPUT_DIRECTORY": args.input_dir,
        "SEARCH_DEPTH": args.search_depth,
        "OUTPUT_DIRECTORY": args.output_dir,
        "ID_LIST": args.id_list
    }

    #Loop over each value
    for key, arg_value in input_dict.items():

        if arg_value == None:

            parsed = False
            while not parsed:

                new_value = input("{0} ({1}): ".format(key, params[key]))

                if new_value == "":
                    parsed=True
                    break

                #If param type is string, just assign directly
                if p_types[key] == str:
                    params[key] = new_value
                    parsed = True

                elif p_types[key] == float:
                    try:
                        params[key] = float(new_value)
                        parsed = True
                    except:
                        print("Error parsing input as float.")
                        continue

                elif p_types(key) == int:
                    try:
                        params[key] = int(new_value)
                        parsed = True
                    except:
                        print("Error parsing input as int.")
                        continue

                elif p_types(key) == list:
                    try:
                        params[key] = new_value.split(',')
                        parsed = True
                    except:
                        print("Error parsing input as comma-separated list.")
                        continue

    if args.out != None:
        outfile = args.out

    else:
        outfile= input("Filename ({0}.param): ".format(params["TARGET_NAME"]))

    parameters.write_params(params, outfile)
    print("Saved {0}".format(outfile))





if __name__=="__main__":
    main()
