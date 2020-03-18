"""Initialize a CWITools parameter file for a new target."""
from cwitools import parameters, utils

import argparse
import warnings

def parse_and_validate(arg, key):
    p_types = parameters.parameter_fields

    if p_types[key] == str:
        return arg, True

    elif p_types[key] == float:
        try:
            arg = float(arg)
            return arg, True
        except:
            return None, False

    elif p_types[key] == int:
        try:
            arg = int(arg)
            return arg, True
        except:
            return None, False

    elif p_types[key] == list:
        try:
            arg = arg.split(',')
            return arg, True
        except:
            return None, False

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
                        type=str,
                        help="Directory to save output files in."
    )
    parser.add_argument('-id_list',
                        type=str,
                        help="Comma separated list of unique identifiers for input files (e.g. image12345 or kb190303_00011)"
    )
    parser.add_argument('-out',
                        type=str,
                        help="Output file name."
    )
    parser.add_argument('-log',
                        type=str,
                        help="Log file to save this command in",
                        def=None
    )
    args = parser.parse_args()

    utils.log_command(sys.argv, logfile=args.log)

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

        #Use interactive mode if no command-line value given
        if arg_value == None:
            parsed = False
            while not parsed:

                new_value = input("{0} ({1}): ".format(key, params[key]))

                if new_value == "":
                    parsed=True
                    break

                else:
                    new_value, parsed = parse_and_validate(new_value, key)
                    if not parsed:
                        print("Error parsing input. Please try again.")
                        continue
                    else:
                        params[key] = new_value

        #Parse command-line arg if given
        else:
            arg_value, parsed = parse_and_validate(arg_value, key)
            if not parsed:
                raise ValueError("Invalid input for {0}".format(key))
            else:
                params[key] = arg_value

    #Get output filename
    if args.out != None:
        outfile = args.out

    else:
        outfile = params["TARGET_NAME"] + ".param"
        new_val = input("Filename ({0}): ".format(outfile))
        if new_val != "":
            outfile = new_val

    #Write to file and notify user
    parameters.write_params(params, outfile)
    print("Saved {0}".format(outfile))





if __name__=="__main__":
    main()
