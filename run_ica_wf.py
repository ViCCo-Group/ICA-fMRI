import os
import sys
from os.path import join, pardir
sys.path.append(pardir)
import bids
from bids import BIDSLayout
from itertools import product, chain
from nipype.pipeline.engine import Workflow
from ica_wf import make_subject_ica_wf

def get_datapaths(bids_layout, outdir, subject, session, run, task, space):
    """
    OUTDATED AND NOT USED ANYOME
    Extract mask and bold file paths for one subject for one task
    of one run for one type of space etc.
    
    Input: BIDSlayout, subject, ...  
    Output: bold filepath, mask filepath and output directory (each as string)
    """
    # check if run and session are present
    run = None if run in ('0','00', '') else run
    session = None if session in ('0','00', '') else session
    # get bold filepaths
    bold_file = bids_layout.get(
                        subject=subject,
                        run=run,
                        session=session,
                        task=task,
                        space=space,
                        extension='nii.gz',
                        suffix='bold',
                        return_type='filename'
                        )
    # check if AROMA was used - if yes, exclude this file
    bold_file = [i for i in bold_file if "AROMA" not in i]
    # get mask filepaths
    mask_file = bids_layout.get(
                        subject=subject,
                        run=run,
                        session=session,
                        task=task,
                        space=space,
                        extension='nii.gz',
                        suffix='mask',
                        return_type='filename'
                        )
    out_dir = join(outdir,
                   f'sub-{subject}',
                   f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-{space}_melodic')
    return bold_file, mask_file, out_dir

def return_datapaths(bids_layout, outdir, subject="all", session="all", run="all", task="all", space="all"):
    """
    OUTDATED AND NOT USED ANYOME
    Check if all data paths or only specific paths are asked for and return full paths.
    
    Input: BIDSlayout
    Optional Input: subject, session, run, task, space
    Output: one file with all bold and mask file paths as tuples
    """
    # TODO: return paths if only one (or more) param is given individually and the others are 'all'
    if all([param == "all" for param in (subject, session, run, task, space)]):
        subject = bids_layout.get(return_type='id', target='subject', desc='preproc')
        session = bids_layout.get(return_type='id', target='session', desc='preproc')
        run = bids_layout.get(return_type='id', target='run', desc='preproc')
        task = bids_layout.get(return_type='id', target='task', desc='preproc')
        space = bids_layout.get(return_type='id', target='space', desc='preproc')
    else:
        subject = subject
        session = session # TODO: for many runs/sessions, check if pybids gives just a number or a full list
        run = run
        task = task
        space = space
    
    # check if run and session are present
    session = '0' if session == [] else session
    run = '0' if run == [] else run
    
    # create all parameter combinations and get their paths
    combinations = list(product(subject, session, run, task, space))
    boldfiles_nested, maskfiles_nested, outdirs_nested = zip(*[
        get_datapaths(bids_layout, outdir, *params) for params in combinations
    ])
    outdirs = list(outdirs_nested)
    boldfiles = [val for sublist in boldfiles_nested for val in sublist]
    maskfiles = [val for sublist in maskfiles_nested for val in sublist]
    
    # create output folders
    for d in outdirs:
        if not os.path.exists(d):
            os.makedirs(d)
    
    return boldfiles, maskfiles, outdirs

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
    boldfile_test = bids_layout.get(
        #scope='derivatives',
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
    maskfile = [i for i in maskfile if "func" in i] # keep files in 'func' folder
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
    elif boldfile == []:
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
    # Check for correct input
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
    # Return datapaths as lists
    boldlist, masklist, outdirlist = get_paths(layout, subjects, output_dir)
    #boldlist, masklist, outdirlist = return_datapaths(layout, output_dir)# --> OUTDATED
    # create output folders
    for d in outdirlist:
        if not os.path.exists(d):
            os.makedirs(d)
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
        #print("boldfile_type", type(boldfile))
        #print("maskfile_type", type(maskfile))
        #print("outdir_type", type(outdir))
        runwf.inputs.inputspec.bold_file = boldfile
        runwf.inputs.inputspec.mask_file = maskfile
        runwf.inputs.inputspec.out_dir = outdir
        runwf.name = join(f'node_{i}')
        #print("runwf.bold_file", runwf.inputs.inputspec.bold_file)
        #print("runwf.mask_file", runwf.inputs.inputspec.mask_file)
        #print("runwf.out_dir", runwf.inputs.inputspec.out_dir)
        #sys.exit()
        
        wf_name = join(f'melodicwf_{i}')
        wf_cloned = runwf.clone(wf_name) # clone sub-workflow with new name
        runwfs.append(wf_cloned)
        i += 1
    
    dataset_wf = Workflow(name='dataset_wf')
    dataset_wf.base_dir = base_dir
    dataset_wf.add_nodes(runwfs)
    print(" DONE \n run workflow ...")
    dataset_wf.run('MultiProc', plugin_args={'n_procs': 30})
