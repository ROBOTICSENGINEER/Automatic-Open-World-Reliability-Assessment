# Automatic Open-World Reliability Assessment


2021 IEEE Winter Conference on Applications of Computer Vision (WACV)


Mohsen Jafarzadeh,  Touqeer Ahmad,  Akshay Raj Dhamija, Chunchun Li, Steve Cruz, Terrance E. Boult


Image classification in the open-world must handle out-of-distribution (OOD) images. Systems should ideally reject OOD images, or they will map atop of known classes and reduce reliability. Using open-set classifiers that can reject OOD inputs can help. However, optimal accuracy of open-set classifiers depend on the frequency of OOD data. Thus, for either standard or open-set classifiers, it is important to be able to determine when the world changes and increasing OOD inputs will result in reduced system reliability. However, during operations, we cannot directly assess accuracy as there are no labels. Thus, the reliability assessment of these classifiers must be done by human operators, made more complex because networks are not 100% accurate, so some failures are to be expected.  To automate this process, herein, we formalize the open-world recognition reliability problem and propose multiple automatic reliability assessment policies to address this new problem using only the distribution of reported scores/probability data. The distributional algorithms can be applied to both classic classifiers with SoftMax as well as the open-world Extreme Value Machine (EVM) to provide automated reliability assessment. We show that all of the new algorithms significantly outperform detection using the mean of SoftMax.


# How to run

### A) Training and saving feature
1. Train EfficientNet-B3 using `train_efficient_b3.py`
2. Extract feautur from EfficientNet-B3 using `save_b3_feature.py`
3. Train EVM using `train_evm.py`
4. Save EVM predivtion using `save_prediction.py`

### B) Batch mode (batch size 100)

1. Run `run_batch_mode.py`
2. Run `plot_batch_1.py`
3. Run `plot_batch_2.py`
4. Run `plot_batch_3.py`

### C) Sliding window mode (window size 100)

1. Run `run_sliding_window_mode.py`
2. Run `plot_sliding_window.py`



# Non overlapping classes

You can see the list of (166 classes) of Imagnet 2010 that are not non-overlapping  with ImageNet 2012 classes in `data/new_166.txt`. Also you can use `data/new_166_dict.json`. 



# License

Copyright (c) 2020 Mohsen Jafarzadeh. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software must display the following acknowledgement: This product includes software developed by Mohsen Jafarzadeh.
4. Neither the name of the Mohsen Jafarzadeh nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY MOHSEN JAFARZADEH "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL MOHSEN JAFARZADEH BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



