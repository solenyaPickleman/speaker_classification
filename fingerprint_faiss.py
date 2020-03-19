import glob
import operator
import faiss 
import pickle
import math
import numpy as np
import time
import torch
import torchaudio

from torch import nn
from functools import reduce
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool

### Define torch parallelism 
torch.set_num_threads(8)

### Define a keyfunction to extract speaker ID from wav file, and get wav file paths from disk
# keyfunc = lambda f :  int(f.split('/')[-1].split('_')[0][1:]) #VCTK
# wavs = glob.glob('/home/sir/voice/VCTK-Corpus/wav48/full_audio/*.wav' )

keyfunc = lambda f : int(f.split('/')[-1].split('-')[0]) #LIBRISPEECH
wavs = glob.glob('/home/sir/voice/LibriSpeech/train-clean-360/full_audio/*.wav' )

###shuffle and build train/test split. 
wavs = np.array(wavs)
np.random.shuffle(wavs)

split = int(len(wavs) * 0.8) 
training, test = wavs[:split], wavs[split:]

### copied  methods for search by pytorch tensor 
### from https://github.com/facebookresearch/faiss/blob/master/gpu/test/test_pytorch_faiss.py

def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)



def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I


### read in WAV files and compute mfcc's via torch
### mfcc's defined : http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

print('start building training mfcc')
start = time.time()

###using the n_mfcc and n_mels from python-speech-features
calc_mfcc = torchaudio.transforms.MFCC(
    n_mfcc=13,
    log_mels=True,
    melkwargs= {
        'n_mels' : 26
    }
)
###zero paddng mfcc to 16 to enable IVFPQ index
calc_mfcc.to('cuda')
pad = nn.ConstantPad1d((0, 16 - 13), 0)
pad.to('cuda')

#for each filename, read the sig from the file, define the key, move the sig to the GPU
#then transpose, pad, numpyify, and add to labels/mfcc_features
labels, mfcc_features = [], []
torch.cuda.empty_cache()
for i,f in enumerate(training):
    sig, sample_rate = torchaudio.load_wav(f)
    key = keyfunc(f)
    sig = sig.to('cuda')
    result = calc_mfcc(sig)[0]
    result = result.transpose(0,1)
    result = pad(result)
    result = result.to('cpu').numpy()
    mfcc_features.extend(result)
    l =  [key for _ in range(len(result))]
    labels.extend(l)
    if i %1000 == 0:
        torch.cuda.empty_cache()
torch.cuda.empty_cache()

print("building training mfcc took : ", time.time()-start  , " seconds")

###convert nfcc_features and labels to the numpy arrays expected by FAISS

start = time.time()
mfcc_features = np.array(mfcc_features, dtype=np.float32)
labels = np.array(labels)
print("formatting labels took : ", time.time()-start  , " seconds")

### Build the FAISS index - in this case, IVFPQ 

d = 16
print('starting training')
start = time.time()

res = faiss.StandardGpuResources()  # use a single GPU

ncentroids = int (4 * math.sqrt (len(labels))) #ncentroids defined on the low end of FAISS guidance. 
gpu_index = faiss.GpuIndexIVFPQ(res, 16,ncentroids, 8,8 , faiss.METRIC_L2)

gpu_index.train(mfcc_features)
gpu_index.add_with_ids(mfcc_features, labels )    # adding vectors to the index w/ ids means that the index handles label lookups
print("building index took : ", time.time()-start  , " seconds")


#save the index to file
index = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(index, "/home/sir/voice/faiss_index_librispeech360.ind")


### Read the test files into GPU memory (and do not bring them to the CPU) to take advantage of GPU querying of torch tensors
start = time.time()

calc_mfcc = torchaudio.transforms.MFCC(
    n_mfcc=13,
    log_mels=True,
    melkwargs= {
        'n_mels' : 26
    }
)
calc_mfcc.to('cuda')
pad = nn.ConstantPad1d((0, 16 - 13), 0)
pad.to('cuda')

testlabels, testfeatures = [], []
torch.cuda.empty_cache()
for f in test:
    sig, sample_rate = torchaudio.load_wav(f)
    key = keyfunc(f)
    sig = sig.to('cuda')
    result = calc_mfcc(sig)[0]
    result = result.transpose(0,1)
    result = pad(result)
    #result = result.to('cpu')
    testfeatures.append(result)
    testlabels.append(key)
torch.cuda.empty_cache()

print("building  test mfcc took : ", time.time()-start  , " seconds")

#########################################################################################################

##IF GPU MEMORY IS A PROBLEM for test/use files
## combine this with the read above and make liberal use of empty cache

start = time.time()
accuracy = []
faiss.GpuParameterSpace().set_index_parameter(gpu_index, "nprobe", 2)

for l, m in tqdm(zip(testlabels, testfeatures), total=len(testlabels)):

    D, I  = search_index_pytorch(gpu_index, m, 5)
    res.syncDefaultStreamCurrentDevice()
    r = torch.flatten(I).cpu().numpy()
    commons = Counter(r).most_common()
    most_likely = commons[0][0]
    accuracy.append(int(most_likely== l))
    
print("Running ",len(testlabels)," queries took ", time.time()-start  , " seconds")
print("Accuracy: ", np.mean(accuracy))



