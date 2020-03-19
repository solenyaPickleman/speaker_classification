# speaker_classification
Speaker Identification via MFCC using pytorch , faiss, and CUDA.  

This method has been tested on two datasets : 
  - VCTK (~44 total hours of audio)
  - Librispeech 360 - (~363 total hours of audio)
using an Intel I7-7700k, 64GB RAM, and an NVIDIA 1080 w/ 8GB VRAM.
```
VCTK results:
  building training mfcc took :  127.95238304138184  seconds
  formatting labels took :  17.94165587425232  seconds
  starting training
  building index took :  103.84340286254883  seconds
  building  test mfcc took :  32.25182032585144  seconds
  100%|███████████████████████████████████████████████████████████████| 8849/8849 [00:23<00:00, 382.26it/s]
  Running  8849  queries took  23.152514457702637  seconds
  Accuracy:  0.9802237540965081

  real	5m7.434s
```
```
Librispeech 360: 
  start building training mfcc
  building training mfcc took :  334.51354789733887  seconds
  formatting labels took :  52.2157666683197  seconds
  starting training
  building index took :  317.4274308681488  seconds
  building  test mfcc took :  96.87221813201904  seconds
  100%|█████████████████████████████████████████████████████████████| 20803/20803 [01:19<00:00, 261.88it/s]
  Running  20803  queries took  79.44146370887756  seconds
  Accuracy:  0.9916358217564775

real	14m45.168s
```

Limiting factors for scaling:
  - CPU limited on reading in wavs - more cores would speed up building the MFCC's 
  - GPU is limited by the amount of VRAM - this took ~ 4GB on the GPU, and bigger indices would need bigger cards.
    - AKA: not an issue for now, but potentially a scaling limit. 
  - RAM is not a huge issue (for now) - Librispeech 360 took < 24GB , so it isn't the limiting factor (for now). 
