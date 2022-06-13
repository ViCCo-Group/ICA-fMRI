import os
import sys
from os.path import join, pardir
sys.path.append(pardir)
import bids
from bids import BIDSLayout
from itertools import product, chain
from nipype.pipeline.engine import Workflow
from ica_wf import make_subject_ica_wf

def get_paths(bids_layout, subject, out_dir):
    """
    Return all datapaths for brain mask files, bold files, and output directories as lists.
    """
    # List with bold filepaths
    boldfile = bids_layout.get(
        scope='derivatives',
        subject=subject,
        extension='nii.gz',
        suffix='bold',
        return_type='filename'
    )
    boldfile = [i for i in boldfile if "AROMA" not in i] # exclude AROMA files
    # List with brainmask filepaths
    maskfile = bids_layout.get(
        scope='derivatives',
        dirname='func',
        subject=subject,
        extension='nii.gz',
        suffix='mask',
        return_type='filename'
    )
    maskfile = [i for i in maskfile if "func" in i] # keep only files in 'func' folder
    # List with output directory paths
    outdirs = []
    for elem in maskfile:
        dir_name = elem.split('/')[-1]
        subj = dir_name.split('_')[0]
        outdir = join(out_dir, subj, f'{dir_name[:-23]}_melodic')
        outdirs.append(outdir)
    # ERROR catching
    if len(boldfile) != len(maskfile): # check lengths
        print("[ERROR] The number of bold files does not match the number of brain mask files.")
        sys.exit()
    elif len(boldfile) != len(outdirs): # check lengths
        print("[ERROR] The number of output directories differs from the number of bold and brain mask files.")
        sys.exit()
    elif boldfile == []: # check if empty
        print("[ERROR] The file lists are empty.")
        sys.exit()
    for bold_elem, mask_elem in zip(boldfile, maskfile): # check names
        boldpath = bold_elem[:-25]
        maskpath = mask_elem[:-23]
        if boldpath != maskpath:
            print("[ERROR] Bold and brain mask files have a different naming convention.")
            print("</path/filename>_desc-preproc_bold.nii.gz")
            print("</path/filename>_desc-brain_mask.nii.gz")
            print("should have same </path/filename>")
            sys.exit()    
    return boldfile, maskfile, outdirs

def description_json(bidsdata_dir):
    """
    Create dataset_description.json and save it to melodic directory.
    """
    dataset_description = {
        "Name": "Melodic - ICA-fMRI",
            "BIDSVersion": "1.4.0",
        "DatasetType": "derivative",
        "PipelineDescription": {
                "Name": "ICA Melodic",
                "Version": "",
                "CodeURL": ""
                },
        "CodeURL": "https://github.com/ViCCo-Group/ICA-fMRI",
        "HowToAcknowledge": "",
        "SourceDatasets": [
            {
                "URL": "",
                "DOI": ""
            }
        ],
        "License": "CC0"
    }
    filename = join(bidsdata_dir, 'derivatives/melodic', 'dataset_description.json')
    with open(filename , 'w', encoding='utf-8') as f:
        json.dump(dataset_description, f, ensure_ascii=False, indent=4)
    return dataset_description

def make_dataset_ica(bidsdata_dir, base_dir, subjects='all', tr=1.5, fwhm=4.):
    """
    From a BIDS dataset, search for all bold and mask files and
    then calculate ICs.
    
    Input:
        bidsdata_dir = '/LOCAL/jzerbe/faces_vs_houses/ds002938'
        base_dir = '/LOCAL/jzerbe/temp_results'
    Optional input:
        subjects = ['01', '03', '15']
        tr = 1.5
        fwhm = 4.0
    Output:
        folder incl. calculated ICs
    """
    # Check input
    if bidsdata_dir in ("", None):
        print("[ERROR] The path to your dataset is missing. Please give an input like -d /path/to/dataset")
        sys.exit()
    if base_dir in ("", None):
        print("[ERROR] The path to your base directory is missing. Please give an input like -b /path/to/base")
        sys.exit()
    print("\nICA Melodic pipeline has started!")
    output_dir = join(bidsdata_dir, 'derivatives', 'melodic')
    hpf = 120. / tr
    print("\n creating BIDS layout ...")
    layout = BIDSLayout(bidsdata_dir, derivatives=True)
    print(" DONE \n get subject IDs ...")
    if subjects == 'all':
        subjects = layout.get_subjects(scope='derivatives', return_type='id')
    else:
        subjects = subjects
    print(" DONE \n get datapaths ...")
    # Return pathways as separate lists
    boldlist, masklist, outdirlist = get_paths(layout, subjects, output_dir)
    #boldlist, masklist, outdirlist = return_datapaths(layout, output_dir)# --> OUTDATED
    # Create output folders
    for d in outdirlist:
        if not os.path.exists(d):
            os.makedirs(d)
    # Create dataset_description.json
    description_json(bidsdata_dir) 
    print(" DONE \n create meta workflow ...")
    # Create meta workflow from single subject workflows
    runwfs = []
    runwf = make_subject_ica_wf()
    runwf.inputs.inputspec.hpf = hpf
    runwf.inputs.inputspec.tr = tr
    runwf.inputs.inputspec.fwhm = fwhm
    runwf.base_dir = base_dir
    i = 1 # iterator to rename sub-workflow

    for boldfile, maskfile, outdir in zip(boldlist, masklist, outdirlist):  
        runwf.inputs.inputspec.bold_file = boldfile
        runwf.inputs.inputspec.mask_file = maskfile
        runwf.inputs.inputspec.out_dir = outdir
        runwf.name = join(f'node_{i}')
        
        wf_name = join(f'melodicwf_{i}')
        wf_cloned = runwf.clone(wf_name) # clone sub-workflow with new name
        runwfs.append(wf_cloned)
        i += 1
    
    dataset_wf = Workflow(name='dataset_wf')
    dataset_wf.base_dir = base_dir
    dataset_wf.add_nodes(runwfs)
    print(" DONE \n run workflow ...")
    dataset_wf.run('MultiProc', plugin_args={'n_procs': 30})
    
    
