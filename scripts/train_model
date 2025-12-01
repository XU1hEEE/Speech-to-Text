"""
train_model.py
==================================================
Description:
    Defines the training configuration, default hyperparameters, and execution
    entry point for the Neural Sequence Decoder. Implements command-line 
    argument parsing for hyperparameter sweeps (ablation studies) and 
    initiates the training process.
    
    Part of the final project for ECE C143A/C243A at UCLA.

Authors:
    Yihe Xu, Injun Choi, Saiya Shah

Date:
    December 1, 2025

Location:
    neural_seq_decoder/scripts/train_model.py
"""


import sys

modelName = 'speechBaseline0'

args = {}
args['outputDir'] = '../src_bk/speech_logs/' + modelName
args['datasetPath'] = '../src_bk/ptDecoder_ctc.pkl'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['nBatch'] = 10000 
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5

# ---- NEW AUGMENTATION SETTINGS ----
args['timeMask'] = -1       # maximum mask length
args['timeMask_N'] = -1     # number of masks
args['channelDrop'] = 0.4     # drop 20% of channels


argv = sys.argv

def get_arg(flag, cast_type, default):
    """Return parsed CLI argument or fallback to default."""
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return cast_type(argv[idx + 1])
    return default

# map the flags your sweep sends into your args dictionary
args['whiteNoiseSD']      = get_arg("--white_std",    float, args['whiteNoiseSD'])
args['constantOffsetSD']  = get_arg("--mean_std",     float, args['constantOffsetSD'])
args['gaussianSmoothWidth'] = get_arg("--gauss_sigma", float, args['gaussianSmoothWidth'])
args['kernelLen']           = get_arg("--gauss_kernel", int,   args['kernelLen'])
args['timeMask']            = get_arg("--mask_max_len", int,   args['timeMask'])
args['timeMask_N']          = get_arg("--mask_n",       int,   args['timeMask_N'])  # NEW
args['channelDrop']         = get_arg("--drop_p",       float, args['channelDrop'])


from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
