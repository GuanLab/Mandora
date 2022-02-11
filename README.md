## Mandora: Machine Learning Reveals Multilevel Interconnections of Joint Damages in Rheumatoid Arthritis 

Mandora is a machine learning approach for quantifying joint damages in rheumatoid arthritis (RA) based on radiographic images.
It ranked first in predicting joint space narrowing in the [RA2 DREAM Challenge](https://www.synapse.org/#!Synapse:syn20545111/discussion/threadId=7376) with high accuracy on held-out testing data.
Beyond high predictive performance, it offers a new way to investigate the cross regulatory relationships among joints and damage types by extracting the characteristic symmetrical patterns in RA.
Please contact (hyangl@umich.edu or gyuanfan@umich.edu) if you have any questions or suggestions.

![Figure1](figure/fig1.png?raw=true "Title")

---

## Installation
Git clone a copy of code:
```
git clone https://github.com/GuanLab/Mandora.git
```
## Required dependencies

* [python](https://www.python.org) (3.6.5)
* [numpy](http://www.numpy.org/) (1.13.3). It comes pre-packaged in Anaconda.
* [tensorflow](https://www.tensorflow.org/) (1.14.0) A popular deep learning package.
```
conda install tensorflow-gpu
```
* [keras](https://keras.io/) (2.2.4) A popular deep learning package using tensorflow backend.
```
conda install keras
```

## Dataset
* [The RA2 DREAM Challenge dataset](https://www.synapse.org/#!Synapse:syn20545111/wiki/597243)

## Code for localizing joints
* code_location

## Code for segmentation of joint space regions and prediction of damage scores
* code_segmentation

## Code for learning symmetry and SHAP analysis 
* code_symmetry




