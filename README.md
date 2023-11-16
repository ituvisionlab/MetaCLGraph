# MetaCLGraph

 This is the official repository of Meta Continual Learning on Graphs with Experience Replay.

 ## Get Started
 
 This repository contains our MetaCLGraph implementation with the baseline models for running on GPU devices. To running in Windows system, please specify the argument ```--replace_illegal_char``` as ```True``` to avoid illegal filename characters (details are in <a href="#Pipeline-Usages">Pipeline Usages</a>). To run the code, the following packages are required to be installed:
 
* python==3.7.10
* scipy==1.5.2
* numpy==1.19.1
* torch==1.7.1
* networkx==2.5
* scikit-learn~=0.23.2
* matplotlib==3.4.1
* ogb==1.3.1
* dgl==0.6.1
* dgllife==0.2.6

 
 ## Dataset Usages
 
 ### Importing the Datasets
 For importing the datasets, use the following command in python:
 
 ```
 from utils import NodeLevelDataset
 dataset = NodeLevelDataset('data_name')
```
 
 where the 'data_name' should be replaced by the name of the selected dataset, which are:
 
 ```
 'CiteSeer-CL'
 'CoraFull-CL'
 'Arxiv-CL'
 'Reddit-CL'
 ```

 ## Pipeline Usages
 
 We provide pipelines for training, validating, and evaluating models with node classification tasks under class incremental scenario. In the following, we provide several examples to demonstrate the usage of the pipelines.
 
 For the experiments, the starting point is the ```train.py``` file, and the different configurations (e.g. the baseline, backbone, etc.), whether to include inter task edges, etc.) are assigned through the keyword arguments of the Argparse module. For example, the following code is used in order to run an experiment with Arxiv-CL dataset with class incremental learning setting while there is not any inter task edges.
 
 ```
 python train.py --dataset Arxiv-CL \
        --method bare \
        --backbone GCN \
        --gpu 0 \
        --ILmode classIL \
        --inter-task-edges False \
        --minibatch False \
 ```

For the baselines with tunable hyper-parameters, the framework provide a convenient way to do grid search over candidate combinations of hyper-parameters. For example, the baseline GEM has two tunable hyper-parameters ```memory_strength``` and ```n_memories```. If the candidate ```memory_strength``` and ```n_memories``` are ```[0.05,0.5,5]``` and ```[10,100,1000]```, respectively, then the command to validate every possible hyper-parameter combination over the validation and test the chosen optimal model on the testing set is as follows:
```
python train.py --dataset Arxiv-CL \
       --gem_args " 'memory_strength': [0.05,0.5,5]; 'n_memories': [10,100,1000] " \
       --method gem \
       --backbone GCN \
       --gpu 0 \
       --ILmode $IL \
       --inter-task-edges $inter \
       --epochs $n_epochs \
```
Some rules to note are: 1. The entire argument ```" 'memory_strength': [0.05,0.5,5]; 'n_memories': [10,100,1000] "``` requires double quotes ``` '' ``` around it. 2. Semicolon ```;``` is required to separate different parameters. 3. Brackets are used to wrap the hyper-parameter candidates. 4. Comma is used to separate the hyper-parameter candidates.
In the example above, all possible nine (three choices for ```memory_strength``` and three choices for ```n_memories```) combinations will be used to set the hyper-parameters.

Since the graphs can be too large to be processed in one batch on most devices, the ```--minibatch``` argument could be specified to be ```True``` for training with the large graphs in mini-batches.
```
 python train.py --dataset Arxiv-CL \
        --method bare \
        --backbone GCN \
        --gpu 0 \
        --ILmode taskIL \
        --inter-task-edges False \
        --minibatch True \
        --batch_size 2000 \
        --sample_nbs True \
        --n_nbs_sample 10,25
 ```
In the above example, besides specifying the ```--minibatch```, the size of each mini-batch is also specified through ```--batch_size```. Moreover, some graphs are extremely dense and will run out the memory even with mini-batch training, which could be addressed through the neighborhood sampling specified via ```--sample_nbs```. And the number of neighbors to sample for each hop is specified through ```--n_nbs_sample```.
There are also other customizable arguments, the full list of which can be found in ```train.py```.

When running the code in Windows system, the following error **OSError: [Errno 22] Invalid argument** may be triggered and could be avoided by specifying the argument ```--replace_illegal_char``` as ```True``` to replace the potential illegal characters with the underscore symbol ```_```. For example,
 ```
 python train.py --dataset Arxiv-CL \
        --method bare \
        --backbone GCN \
        --gpu 0 \
        --ILmode classIL \
        --inter-task-edges False \
        --minibatch False \
        --replace_illegal_char True 
 ```

### Modifying the train-validation-test Splitting

The splitting can be simply specified via the arguments when running the experiments. In our implemented pipeline, the corresponding arguments are the validation and testing ratios. For example,

```
python train.py --dataset Arxiv-CL \
        --method bare \
        --backbone GCN \
        --gpu 0 \
        --ILmode taskIL \
        --inter-task-edges False \
        --minibatch False \
        --ratio_valid_test 0.4 0.4
```

The example above set the data ratio for validation and testing as 0.4 and 0.4, and the training ratio is automatically calculated as 0.2.

 ## Acknowledgement
 The construction of our repository also benefits from existing repositories on both continual learning and continual graph learning. The pipeline and the benchmark models were adapted from [CGLB](https://github.com/QueuQ/CGLB). The construction of the datasets also benefits from several existing databases and libraries. The construction of the datasets uses the datasets and tools from OGB and DGL. 

We sincerely thank the authors of these works for sharing their code and helping developing the community.
