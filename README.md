# A general deep learning method for computing molecular parameters of viscoelastic constitutive model by solving an inverse problem
___
 Link to our paper [[Arxiv](https://arxiv.org/abs/18.043)]

# Requirements
1. It is encouraged to create a new conda environment for running this code\
    `conda env create -f environment.yml`\
    `conda activate inv_learning`

2. If a similar environment is deployed for running this code, ensure these packages are included in your environment
   * Python (version >= 3.6)
   * PyTorch (version >= 0.4.0)
   * Numpy
   * Matplotlib


# Description
The following directories contain the data of our implementation:
* `gxymodel/` contains an example of our pre-trained DNN model, which can be loaded for inverse learning.
* `exp_data/` contains processed experimental data for high molecular weight polyacrylamide aqueous solution
* `log_data/` contains the processed data of Rolie-Poly. Each directory of this folder contains a dataset sampled uniquely (One was sampled randomly from linear space and the other one was sampled in logarithmic space under the predetermined parameter space). 
* `property/` will save the found parameter in a standard file(constitutiveProperties) which can be used to compute the corresponding stresses with OpenFOAM. 
* `exp/` is used to save the running results of the experiments.


# Quick Start
1. Train a DNN model that can represent a constitutive model:\
```python -u fast_forwordDnn.py --hidden-layer 6 --neuron 192 --lr 0.001 --exp-name mix_Random_10000data --dataset 10000_targetRange_logRandom  --exp-itr 0811 --n-epoch 1000000  --data 5000 --testdata 10000 --batch-size 1000```
* `dataset` defines which dataset is used to evaluate the quality of DNN.
* `testdata` defines the start number to slice a test dataset (There are 11381 sets of data in `10000_targetRange_logRandom`. All data whose index is over the value of `testdata` will be taken as test data) 
* `data` defines the number of data that will be loaded for training in each dataset. It should be noted that the number of data loaded from each dataset is equal.

2. Inverse Learning for lambdaDNA:\
```python -u fast_inv_grad_sgd.py --n-epoch 30000 --layer 6 --neuron 192 --seeds 10 --checkpoint fast_G_xy0.0002529911871533841 --testset lambdaDNA --lr 0.4 --order 0```
* `order` indicates the index of data to load stress tensors.
* `seeds` indicates the number of optimizing processes that will run parallelly. (According to our experience, 10 will be enough to find the optimal set of parameter)

3. Inverse Learning for high molecular weight polyacrylamide aqueous solution:\
`python -u fast_inv_grad_expdata.py --data-file 48c_fast --checkpoint fast_G_xy0.0002529911871533841 --seeds 100`
* `data-file` includes '48c', '97c' and '150c', which is the processed experiment data.

# Contact
Minghui Ye (yeminghui1@gmail.com)
