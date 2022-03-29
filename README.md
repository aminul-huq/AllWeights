# AllWeights


This is a public repository for different weight assignment schemes in multi-task learning systems. Using this library users can create objects of different weight assignment schemes and use it directly for the loss value.

# Requirements
PyTorch > 1.9 Python > 3.7

# Installation
Clone this library to your project using the following command
```
https://github.com/aminul-huq/AllWeights.git
```

# Weight assignment techniques

| Method name        |   Description    | 
| -------------  |:--------------------:| 
| equal.py       | Each loss will have equal weights                    | 
| random.py           | Each loss will have random weights                |  
| dwa.py          | End-to-End Multi-Task Learning with Attention [CVPR 2019]                  |   
| uncertainty.py      | Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics [CVPR 2018]                 |
| reu.py   | Auxiliary tasks in multi-task learning. arXiv preprint arXiv:1805.06334, 2018.                |  


N.B. DWA method needs to be updated.
# Example

```
from AllWeights import *

w = Equal(3)

Loss = w([Task1_loss, Task2_loss, Task3_loss])

```
