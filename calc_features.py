import os
import sys
import getopt
import numpy as np
import pandas as pd
from nilearn.image import load_img, threshold_img, math_img, resample_to_img
from nilearn.masking import intersect_masks, apply_mask
from scipy.ndimage.morphology import binary_erosion
from scipy.signal import periodogram
from os.path import join, pardir
sys.path.append(pardir)
from bids import BIDSLayout
from tqdm import tqdm

# Helper functions
def get_comps(metainfo_dict):
    """Get comparison array."""
    mixmat = np.loadtxt(join(metainfo_dict['fullpath'], 'melodic_mix'))
    ica_nii_f = join(metainfo_dict['fullpath'], 'melodic_IC.nii.gz')
    comps_arr = load_img(ica_nii_f).get_fdata()
    return mixmat, comps_arr

def get_edge_mask(metainfo_dict, ds_layout):
    """Retrieve fmriprep/func brainmask files & calculate edge mask."""
    brainmask_f_temp = ds_layout.get(
        scope='derivatives',
        return_type='filename',
        subject=metainfo_dict['subject'],
        session=metainfo_dict['session'],
        run=metainfo_dict['run'],
        task=metainfo_dict['task'],
        space=metainfo_dict['space'],
        desc='brain',
        suffix='mask',
        extension='nii.gz'
    )
    brainmask_f = brainmask_f_temp[0]
    if metainfo_dict['space'] == 'T1w':
        csf_anat_f_temp = ds_layout.get(
                #scope='fmriprep', -> TODO: seems not to work
                return_type='filename',
                subject=metainfo_dict['subject'],
                label='CSF',
                suffix='probseg',
                extension='nii.gz'
        )
        csf_anat_f = csf_anat_f_temp[0]
    else:
        csf_anat_f_temp = ds_layout.get(
                #scope='fmriprep', -> TODO: seems not to work
                return_type='filename',
                subject=metainfo_dict['subject'],
                space=metainfo_dict['space'],
                label='CSF',
                suffix='probseg',
                extension='nii.gz'
        )
        csf_anat_f = csf_anat_f_temp[0]
    csf_func = threshold_img(
        resample_to_img(csf_anat_f, brainmask_f, interpolation='linear'),
        threshold=1.
    )
    brainmask = load_img(brainmask_f).get_fdata()
    mask_img = math_img('img1 - img2', img1=brainmask_f, img2=csf_func)
    mask_arr = mask_img.get_fdata()
    # worked okayish with erosion iterations=2
    edgefrac_thickness = int(2)
    ero_mask = binary_erosion(mask_arr, iterations=edgefrac_thickness).astype(int)
    edgemask = mask_arr - ero_mask
    return edgemask.astype(bool), brainmask.astype(bool)

def get_gmfiles(metainfo_dict, ds_layout):
    """ ... """
    bmask_f = ds_layout.get(
        scope='derivatives',
        return_type='filename',
        subject=metainfo_dict['subject'],
        session=metainfo_dict['session'],
        run=metainfo_dict['run'],
        task=metainfo_dict['task'],
        space=metainfo_dict['space'],
        desc='brain',
        suffix='mask',
        extension='nii.gz'
    )
    aseg_f = ds_layout.get(
        scope='derivatives',
        return_type='filename',
        subject=metainfo_dict['subject'],
        session=metainfo_dict['session'],
        run=metainfo_dict['run'],
        task=metainfo_dict['task'],
        space=metainfo_dict['space'],
        desc='aseg',
        suffix='dseg',
        extension='nii.gz'
    )
    # aseg2gm
    gm_left = math_img('img == 3', img=aseg_f)
    gm_right = math_img('img == 42', img=aseg_f)
    gm = intersect_masks([gm_left, gm_right], threshold=0, connected=False)
    return bmask_f[0], gm

def get_zstat_and_power(comp_i, ds_layout, metainfo_dict):
    """Retrieve melodic thresh_zstat file for gray matter feature."""
    comp_fs = ds_layout.get(
        scope='melodic',
        subject=metainfo_dict['subject'],
        return_type='filename',
        suffix=f'zstat{comp_i}',
        extension='nii.gz'
    )
    comp_f = [s for s in comp_fs if metainfo_dict['directory'] in s]
    power_list = ds_layout.get(
        scope='melodic',
        subject=metainfo_dict['subject'],
        return_type='filename',
        suffix=f'f{comp_i}',
        extension='txt'
    )
    power_f = [s for s in power_list if metainfo_dict['directory'] in s]
    return comp_f[0], power_f[0]

def calc_edgefrac(comp_arr, edgemask, brainmask):
    """Calculate edge fraction."""
    return np.absolute(comp_arr[edgemask]).sum() / np.absolute(comp_arr[brainmask]).sum()

def calc_hfc(timeseries, tr=1.5):
    """Calculate high frequency content for time series data. Tr can generally mean sampling rate in seconds."""
    nf = (1. / tr) * .5  # nyquist
    freqs, power = periodogram(timeseries, fs=1. / tr)
    relcumsum = np.cumsum(power) / power.sum()
    freqind = np.argmin(np.absolute(relcumsum - .5))
    hfc = freqs[freqind] / nf
    return hfc

def calc_gm_prop(comp_f, bmask_f, gm):
    """
    Calculate the proportion of the (active) gray matter voxels of a volume
    compared to the whole number of gray voxels in the brain.
    """
    # count significant voxels in gray matter
    comp_gm = apply_mask(comp_f, gm)
    nsig_gm = np.sum(comp_gm > 0.)
    # count significant voxels in whole brain
    comp_brain = apply_mask(comp_f, bmask_f)
    nsig_brain = np.sum(comp_brain > 0.)
    # return ratio
    gm_prop = nsig_gm / nsig_brain
    return gm_prop

def calc_pss(power_f):
    """
    Calculate the power spectrum skewness (pss) by comparing the
    lower third of an IC's frequency with whole power spectrum.
    """
    power_arr = np.loadtxt(power_f, dtype=np.float)
    # Check how much of the power lies (roughly) in the 'lower third'
    power_ic = np.trapz(power_arr[:round(len(power_arr)*0.35)])
    power_all = np.trapz(power_arr[:len(power_arr)])
    # Compare to overall power of IC
    pss = power_ic/power_all
    return pss

# Main function feature calculation
def calculate_features(bidsdata_dir):
    """Get dict with calculated features for each melodic run."""
    melodic_base_dir = join(bidsdata_dir, 'derivatives', 'melodic')
    print("  creating BIDS layout ...")
    ds_layout = BIDSLayout(bidsdata_dir, derivatives=True)
    # TODO: 'melodic' needs dataset_description.json or melodic_entities is empty
    print("  get melodic directory names ...")
    melodic_entities = ds_layout.get(scope='melodic', return_type='filename', suffix='IC', extension='nii.gz')
    print("  [Sanity check] melodic_entities length: ", len(melodic_entities))
    results_dicts = []
    for entity in tqdm(melodic_entities, desc='  iterating over runs'):
        # Cumbersome workaround to get correct filenames --> TODO: better filenaming with DataSink!
        melodic_dir_split = entity.split('/')
        dir_name = melodic_dir_split[-2]
        metainfo_split = dir_name.split('_')
        # Check if entity is present for this run
        subject = [s[4:] for s in metainfo_split if "sub-" in s]
        session = [s[4:] for s in metainfo_split if "ses-" in s]
        task = [s[5:] for s in metainfo_split if "task-" in s]
        run = [s[4:] for s in metainfo_split if "run-" in s]
        space = [s[6:] for s in metainfo_split if "space-" in s]
        # Create dict with run info
        metainfo_dict = {
            'subject': None if subject in ([], ['None']) else subject[0],
            'session': None if session in ([], ['None']) else session[0],
            'task': None if task in ([], ['None']) else task[0],
            'run': None if run in ([], ['None']) else run[0],
            'space': None if space in ([], ['None']) else space[0],
            'directory': dir_name,
            'fullpath':'/'.join(melodic_dir_split[:-1])
        }
        mixmat, comps_arr = get_comps(metainfo_dict)
        edgemask, brainmask = get_edge_mask(metainfo_dict, ds_layout)
        bmask_f, gm = get_gmfiles(metainfo_dict, ds_layout)
            
        for comp_i in range(mixmat.shape[-1]):
            results_dict = {
                'subject': metainfo_dict['subject'],
                'session': metainfo_dict['session'],
                'task': metainfo_dict['task'],
                'run': metainfo_dict['run'],
                'space': metainfo_dict['space'],
                'directory': metainfo_dict['directory'],
                'fullpath': metainfo_dict['fullpath']
            }
            comp_arr = comps_arr[:, :, :, comp_i]
            comp_ts = mixmat[:, comp_i]
            # Get IC stat file for gm prop
            comp_f, power_f = get_zstat_and_power((comp_i + 1), ds_layout, metainfo_dict)
            # Calculate edge fraction
            results_dict['edgefrac'] = calc_edgefrac(comp_arr, edgemask, brainmask)
            # Calculate high frequency content
            results_dict['hfc'] = calc_hfc(comp_ts)
            # Calculate gray matter proportion
            results_dict['gmp'] = calc_gm_prop(comp_f, bmask_f, gm)
            # Calculate power spectrum skewness
            results_dict['pss'] = calc_pss(power_f)
            results_dicts.append(results_dict)
    return results_dicts

if __name__ == "__main__":
    arg_dataset = ""
    arg_name = ""
    arg_help = """
    FILENAME
        {0}
    
    DESCRIPTION
        Calculate different features from melodic data. These features include the
        high frequency content (hfc), edge fraction, etc.
        Results are saved as pandas dataframe.
        
    SYNTAX
        python {0} [--directory] <path> [--name] <string>
    
    ARGUMENTS
        --directory -d
            Path to the BIDS dataset.
            
        --name -n
            Name of dataset. Used for result, i.e. df_features_<ds-name>.csv
            
        --help -h
            
    ------------------------- EXAMPLE -------------------------------
    
    python3 {0} -d /path/to/dataset/ds00xxxx -n ds-name

    """.format(sys.argv[0])
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:n:", ["help", "directory=", "name="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-d", "--directory"):
            if not os.path.exists(arg):
                print("[ERROR] Your dataset directory path cannot be found. Please enter a valid path.")
                sys.exit()
            arg_directory = arg
        elif opt in ("-n", "--name"):
            arg_name = arg
    print("\n## Start feature calculation. ##\n")
    results_dicts = calculate_features(arg_directory)
    results_df_all = pd.DataFrame(results_dicts)
    # TODO: Make results path adaptable when done with own calculations
    results_df_all.to_csv(f'/LOCAL/jzerbe/code/ICA-fMRI/results/df_features_{arg_name}.csv')
    print("\n## Feature calculation finished. ##")