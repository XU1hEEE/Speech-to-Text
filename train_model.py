import sys

modelName = 'speechBaseline4'

args = {}
args['outputDir'] = 'C:/Users/greys/Documents/neural_seq_decoder/outputs/logs/speech_logs/' + modelName
args['datasetPath'] = 'C:/Users/greys/Documents/neural_seq_decoder/notebooks/.ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128 #64
args['lrStart'] = 0.05 #0.02
args['lrEnd'] = 0.02
args['nUnits'] = 256 #1024
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.2 #0.4

args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False #True
args['l2_decay'] = 1e-5

# ---- NEW AUGMENTATION SETTINGS ----
args['timeMask'] = -1       # maximum mask length
args['timeMask_N'] = -1     # number of masks
args['channelDrop'] = 0.4     # drop 20% of channels
# args['useTimeShift'] = True   # optional flags
# args['useTimeMask'] = True
# args['useChannelDrop'] = True
# ----------------------------------

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


"""
.\.venv\Scripts\Activate.ps1
uv run scripts/train_model.py
 uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
"""

#train results: 
"""
1. baseline4: cer .238; 34 minutes
2. (whiteNoiseSD: 0.8, gaussianSmoothWidth: 2.0, channelDrop: 0.1) batch 9900, ctc loss: 0.899751, cer: 0.229458, time/batch:   0.210
3. (whiteNoiseSD: 0.8, gaussianSmoothWidth: 2.0, channelDrop: 0.2) batch 9900, ctc loss: 0.852928, cer: 0.224671, time/batch:   0.207
"""



 #python "C:\Users\greys\Documents\neural_seq_decoder\scripts\run_augmentation_sweep.py"
