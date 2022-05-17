import os
import sys
import getopt
from os.path import pardir
sys.path.append(pardir)
from run_ica_wf import make_dataset_ica

def parse_args_and_start_ica(argv):
    """
    Receive the full pathways for the BIDS dataset, the base directory, and the output
    directory as inputs, then run the ICA workflow with them.
    
    Private Example (how to call the whole file from shell):
    python3 ica_main.py -d /LOCAL/jzerbe/faces_vs_houses/ds002938 -b /LOCAL/jzerbe/temp_results -o /LOCAL/jzerbe/temp_results/melodic
    """
    arg_dataset = ""
    arg_base = ""
    arg_output = ""
    arg_help = "\n    {0} -d <dataset> -b <base> -o <output> \n\n    Please enter the full pathway to your BIDS dataset, the base directory,\n    and the destination for the output.\n\n    Example:\n        -d '/path/to/dataset/ds00xxxx'\n        -b '/base/directory'\n        -o '/path/to/output/directory'\n\n".format(argv[0]) 
    try:
        opts, args = getopt.getopt(argv[1:], "hd:b:o:", ["help", "dataset=", "base=", "output="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-d", "--dataset"):
            arg_dataset = arg
        elif opt in ("-b", "--base"):
            arg_base = arg
        elif opt in ("-o", "--output"):
            arg_output = arg
    make_dataset_ica(arg_dataset, arg_base, arg_output)

if __name__ == "__main__":
    parse_args_and_start_ica(sys.argv)