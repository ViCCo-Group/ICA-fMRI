import numpy as np
from nipype.interfaces.fsl.maths import TemporalFilter
from nipype.interfaces.fsl.model import MELODIC
from nipype.interfaces.fsl.preprocess import SUSAN
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.pipeline.engine import Workflow, Node

def calc_susan_thresh(boldfile, maskfile, timeax=0, median_factor=.75):
    """
    Calculate the median value within brainmask and multiply with fixed
    factor to get an estimate of the contrast between background and brain
    for FSL's SUSAN.
    """
    from nilearn.masking import apply_mask
    import numpy as np
    data = apply_mask(boldfile, maskfile)
    med = np.median(data.mean(axis=timeax))
    del data  # suspect memory leak
    return med * median_factor

def make_subject_ica_wf():
    """
    Example Inputs:
        wf.inputs.inputspec.bold_file = '/../preproc_bold.nii.gz'
        wf.inputs.inputspec.mask_file = '/../brain_mask.nii.gz'
        wf.inputs.inputspec.tr = 1.5
        wf.inputs.inputspec.hpf = 120.0/tr
        wf.inputs.inputspec.fwhm = 4.0
        wf.inputs.inputspec.out_dir = '/results/melodic'

        wf.base_dir = '/..'
    
    Output:
        nipype workflow = combining precessing (smoothing/temporal filtering)
        and ICA for one functional run

    TODO: give TR as input (to this function) or infer from data?
    """
    # create input spec
    inputspec = Node(
        IdentityInterface(
            fields=['bold_file',
                    'mask_file',
                    'tr',
                    'hpf',
                    'fwhm',
                    'out_dir']
                    ),
            name="inputspec")
    
    # create node for smoothing
    calcthresh = Node(
        Function(
            function=calc_susan_thresh, input_names=['boldfile', 'maskfile'], 
            output_names=['smooth_thresh']),
            name='calcthresh'
    )
    susan = Node(SUSAN(), name='susan') # requires 
    # ... temporal filtering
    tfilt = Node(TemporalFilter(), name='tfilt')
    # ... ICA
    melodic = Node(MELODIC(out_all=True, no_bet=True, report=True), name='melodic')
    
    # connect nodes in workflow
    wf = Workflow(name='melodicwf')
    wf.connect([
        (inputspec, calcthresh, [('bold_file', 'boldfile'),
                                 ('mask_file', 'maskfile')]),
        (inputspec, susan, [('bold_file', 'in_file'),
                            ('fwhm', 'fwhm')]),
        (inputspec, tfilt, [('hpf', 'highpass_sigma')]),
        (inputspec, melodic, [('mask_file', 'mask'),
                              ('out_dir', 'out_dir'),
                              ('tr', 'tr_sec')]),
        (calcthresh, susan, [('smooth_thresh', 'brightness_threshold')]),
        (susan, tfilt, [('smoothed_file', 'in_file')]),
        (tfilt, melodic, [('out_file', 'in_files')])
    ])
    return wf
