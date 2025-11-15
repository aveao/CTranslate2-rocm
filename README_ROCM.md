# ROCm CT2

<div align="center">

[Upstream README](README.md) | **ROCm Install Guide**
</div>

This is a fork of https://github.com/arlo-phoenix/CTranslate2-rocm with:
- changes from https://github.com/mmis1000/CTranslate2-rocm/tree/rocm-windows-pr applied for ROCm 7 support (-> though ROCm 7 isn't working perfectly at the moment)
- changes from https://github.com/justinkb/CTranslate2-rocm/tree/4.6.0-rocm applied to go up to CTranslate2 4.6.0, alongside further fixes from that repo

That was then rebased onto CTranslate2 4.6.1, with some improvements/fixes on build instructions by me (@aveao). This is my first rodeo with ROCm, don't expect me to have done anything "right".

## Install Guide

### Directly with docker

`docker build --file docker/Dockerfile-rocm --tag ctranslate2:4.6.1-rocm6 --build-arg ROCM_ARCH=SET_ME .`

If you want pytorch also:

`docker build --file docker/Dockerfile-rocm --tag ctranslate2:4.6.1-rocm6-pytorch --build-arg ROCM_ARCH=SET_ME --build-arg RUNNER_IMAGE="rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1" .`

Value for `ROCM_ARCH` can be found through `rocminfo | grep gfx`.

When running, pay attention to https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html

#### ROCm 7

ROCm 7 isn't working right now, haven't yet figured out why.

You can play around with it with this command if you want:

`docker build --file docker/Dockerfile-rocm --tag ctranslate2:4.6.1-rocm7 --build-arg ROCM_ARCH=SET_ME --build-arg BUILDER_IMAGE="rocm/dev-ubuntu-24.04:7.1-complete" .`

### Manual setup

These install instructions are for https://hub.docker.com/r/rocm/dev-ubuntu-24.04 (tested for `:6.4.4-complete`). They should mostly work for system installs as well, but then you'll have to change install directories and make sure all dependencies are installed. This also installs things system-wide, you might want to change that if not in docker.

After following the guide in https://hub.docker.com/r/rocm/pytorch for env setup, running inside the container:

```bash
git clone https://github.com/aveao/CTranslate2-rocm.git --recurse-submodules
apt update
apt install cmake libomp-20-dev
cd CTranslate2-rocm
#export ROCM_ARCH=gfx1201 #optionally set this only to your ROCm arch to speed up compiling. You can find it with rocminfo | grep gfx
CLANG_CMAKE_CXX_COMPILER=amdclang++ CXX=amdclang++ HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DWITH_MKL=OFF -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=$ROCM_ARCH -DWITH_DNNL=OFF -DBUILD_TESTS=ON -DWITH_CUDNN=ON
cmake --build build -- -j
cd build
make install
ldconfig
cd ../python
pip install -r install_requirements.txt
python setup.py bdist_wheel
pip install dist/*.whl
```

## Benchmarks

### faster-whisper

faster-whisper 1.2.1, ctranslate 4.6.1, whisper-3-large-turbo, ROCm 6.4.4

GPU running in docker, CPU running on system.

Audio Duration: 734.16s (mp3). Noisy over-the-air recording of a ham radio communication (few points without speech). No VAD applied.

**language unset (de), no batching, beam_size=5:**

- Radeon RX 9070 XT (FP16): 26.966s
- Ryzen 7 9700X (performance): 85.753s
- Ryzen 7 9700X (power save): 96.006s

**language unset (de), batch_size=16:**

- Radeon RX 9070 XT (FP16): 19.109s
- Ryzen 7 9700X (performance): 48.371s

**language set (de), batch_size=16:**

- Radeon RX 9070 XT (FP16): 18.380s
- Ryzen 7 9700X (performance): 46.560s

## Running tests / debugging issues

(anything below is from the original forks, I haven't tested them)

### Tests

In CT2 project root folder:

```bash
./build/tests/ctranslate2_test ./tests/data/ --gtest_filter=*CUDA*:-*bfloat16*
```
for me only some int8 test failed (I think that test shouldn't even be run for CUDA, but didn't check too deeply. The guard is from CT2 itself so it's supposed to fail)

### Checking that all libraries are found

`ld -lctranslate2 --verbose` (ignore warnings, only important thing is that it doesn't find link errors)

### BF16 issues

This fork just commented out everything related to bf16. I think an implicit conversion operator from  `__hip_bfloat16` to `float` is missing

example error with bf16 enabled:
```cpp
CTranslate2/src/cuda/primitives.cu:284:19: error: no viable conversion from 'const __hip_bfloat16' to 'const float'
  284 |       const float score = previous_scores[i];
```

Other than that I **won't** be adding FA2 or AWQ support. It's written with assembly for cuda and it isn't helpful at all for my use case (whisper). Otherwise on this older commit (besides bf16) this fork is feature complete, so I might look into cleaning it up and possibilities of disabling these for ROCm on master for upstreaming. But I'll only do that **after BF16 gets proper support** since this discrepancy adds way too many different code paths between ROCm and CUDA. Other than that conversion worked quite well, for the majority of the project I only had to change a couple defines. Only the `conv1d OP` required a custom implementation for MIOpen (hipDNN isn't maintained anymore).

## Tested libraries

### faster-whisper
```bash
pip install faster-whisper

#1.0.3 was the most recent version when I made this, so try testing that one first if a newer one doesn't work
#pip install faster-whisper==1.0.3
```

I included a small benchmark script in this CT2 fork. You need to download a test file from the faster whisper repo
```bash
wget -P "./tests/data" https://github.com/SYSTRAN/faster-whisper/raw/master/tests/data/physicsworks.wav 
```

Then you should be able to run 

```bash
python faster_whisper_bench.py
```
This per default does just one testrun with the medium model. I'm getting around `10.9-11.0s` on my RX6800 (with model loading included `13.7-13.8s`).


### whisperX

System dependency is just ffmpeg. Either use your system package manager or with conda `conda install conda-forge::ffmpeg`

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1 --force-reinstall
pip3 install transformers pandas nltk pyannote.audio==3.1.1 faster-whisper==1.0.1 -U
pip3 install whisperx --no-deps
```
Python dependencies are a mess here since versions aren't really pinned and the image doesn't come with `torchaudio`. The commands above worked for me though, but will take a while since this reinstalls all python dependencies.

For running you can use its great cli-tool by just using `whisperx path/to/audio` or running my little bench script for the `medium` model.

```bash
python whisperx_bench.py
```

this took around `4.1s` with language detection and around `3.94s` without.

If you do get it running it's pretty fast. I excluded model load since that one takes quite a while. With model load it was only slightly faster than faster_whisper, but I think that's connected with the bunch of version conflicts I had. The main advantage of `whisperx` is its great feature set (Forced Alignment, VAD, Speaker Diarization) and the cli-tool (lots of output options), so do try and get it running it's worth it.