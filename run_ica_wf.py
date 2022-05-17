import os
import sys
from os.path import join, pardir
sys.path.append(pardir)
from bids import BIDSLayout
from itertools import product, chain
from nipype.pipeline.engine import Workflow
from ica_wf import make_subject_ica_wf

def get_datapaths(bids_layout, subject, session, run, task, space, outdir):
    """
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
                   f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-{space}-melodic')
    return bold_file, mask_file, out_dir

def return_datapaths(bids_layout, outdir, subject="all", session="all", run="all",
               task="all", space="all"):
    """
    Check if all data paths or only specific paths are asked for and return full paths.
    
    Input: BIDSlayout
    Optional Input: subject, session, run, task, space
    Output: one file with all bold and mask file paths as tuples
    """
    # TODO: return paths if only one (or more) param is given individually and the others are 'all'
    if all([param == "all" for param in (subject, session, run, task, space)]):
        subject = bids_layout.get(return_type='id', target='subject', desc='preproc')
        session = bids_layout.get(return_type='id', target='session', desc='preproc')
        run = bids_layout.get(return_type='id', target='session', desc='preproc')
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
        get_datapaths(bids_layout, *params, outdir) for params in combinations
    ])
    outdirs = list(outdirs_nested)
    boldfiles = [val for sublist in boldfiles_nested for val in sublist]
    maskfiles = [val for sublist in maskfiles_nested for val in sublist]
    
    # create output folders
    for d in outdirs:
        if not os.path.exists(d):
            os.makedirs(d)
    
    return boldfiles, maskfiles, outdirs

def make_dataset_ica(bidsdata_dir, base_dir, out_dir, tr=1.5, hpf=80., fwhm=4.):
    """
    From a BIDS dataset, search for all bold and mask files and
    then calculate ICs.
    
    Input:
        bidsdata_dir = '/LOCAL/jzerbe/faces_vs_houses/ds002938'
        base_dir = '/LOCAL/jzerbe/temp_results'
        out_dir = '/LOCAL/jzerbe/temp_results/melodic'  # TODO: change to bidsdata_dir + /melodic?
    Optional input:
        tr = 1.5
        hpf = 80. # 120./TR
        fwhm = 4.0
    Output:
        folder incl. calculated ICs
    """
    # The 'layout' function can throw error if derivatives are not saved inside
    # the BIDS dataset as a folder called 'derivatives', and also if the
    # json description file does not include ... [to-be-added]
    
    layout = BIDSLayout(bidsdata_dir, derivatives=True) 
    boldlist, masklist, outdirlist = return_datapaths(layout, out_dir)
    
    runwfs = []
    runwf = make_subject_ica_wf()
    runwf.inputs.inputspec.hpf = hpf
    runwf.inputs.inputspec.tr = tr
    runwf.inputs.inputspec.fwhm = fwhm
    runwf.base_dir = base_dir
    i = 1 # iterator to rename workflow

    for boldfile, maskfile, outdir in zip(boldlist, masklist, outdirlist):       
        runwf.inputs.inputspec.bold_file = boldfile
        runwf.inputs.inputspec.mask_file = maskfile
        runwf.inputs.inputspec.out_dir = outdir
        runwf.name = join(f'node_{i}')
        
        wf_name = join(f'melodicwf_{i}')
        wf_cloned = runwf.clone(wf_name) # clone workflow with new name
        runwfs.append(wf_cloned)
        i += 1
    
    dataset_wf = Workflow(name='dataset_wf')
    dataset_wf.base_dir = base_dir
    dataset_wf.add_nodes(runwfs)
    
    dataset_wf.run('MultiProc', plugin_args={'n_procs': 30})
