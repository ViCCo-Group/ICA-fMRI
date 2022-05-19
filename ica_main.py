import os
import sys
import getopt
from os.path import pardir
sys.path.append(pardir)
from run_ica_wf import make_dataset_ica

def parse_args_and_start_ica(argv):
    """
    Receive the full pathways for the BIDS dataset and base directory as inputs,
    then run the ICA workflow.
    
    Private Example (how to call the whole file from shell):
    python3 ica_main.py -d /LOCAL/jzerbe/faces_vs_houses/ds002938 -b /LOCAL/jzerbe/temp_results
    """
    # TODO: add optional input (tr, hpf, fwhm) & (subject, session, run, etc.)
    arg_dataset = ""
    arg_base = ""
    arg_tr = float(1.5)
    arg_fwhm = float(4.)
    arg_help = """
    FILENAME
        {0}
    
    DESCRIPTION
        Perform Independent Component Analysis on a preprocessed dataset.
        The outputs are calculated ICs saved at '/../derivatives/melodic'
        of the called dataset. Please enter the FULL pathway to your BIDS
        dataset and base directory.
        
    SYNTAX
        python {0} [--dataset] <path> [--base] <path>
        [[--tr] <float>] [[--fwhm] <float>]
    
    ARGUMENTS
        --dataset -d
            Path to the BIDS dataset.
            
        --base -b
            Path to base directory. The working directory will be stored here.
        
        --tr -t
            (optional)
            Temporal resolution. Defaults to 1.5
        
        --fwhm -f 
            (optional)
            Full-width-half-maximum. Defaults to 4.0
            
    ------------------------- EXAMPLE -------------------------------
    
    python3 {0} -d /path/to/dataset/ds00xxxx -b /path/to/base
    
    """.format(argv[0]) 
    try:
        opts, args = getopt.getopt(argv[1:], "hd:b:t:f:", ["help", "dataset=", "base=", "tr=", "fwhm="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-d", "--dataset"):
            if not os.path.exists(arg):
                print("[ERROR] Your dataset path cannot be found. Please enter a valid path.")
                sys.exit()
            arg_dataset = arg
        elif opt in ("-b", "--base"):
            if not os.path.exists(arg):
                print("[ERROR] Your base directory cannot be found. Please enter a valid path.")
                sys.exit()
            arg_base = arg
        elif opt in ("-t", "--tr"):
            arg_tr = float(arg)
        elif opt in ("-f", "--fwhm"):
            arg_fwhm = float(arg)
    
    make_dataset_ica(arg_dataset, arg_base, arg_tr, arg_fwhm)

if __name__ == "__main__":    
    parse_args_and_start_ica(sys.argv)