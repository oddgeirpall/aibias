# AI Bias Mitigation


Package that includes implementations on various metrics and algorithms
that deal with mitigating biases in datasets and models.

## Installation

The package can be installed via pip

`pip install AIBias-Oddgeir`

You can also build form the source code by cloning the repo and running

`make`

## Usage

To use this package you first need to create a dataset object (`aibias.dataset.Dataset`).  
For that you need a `pandas.DataFrame`, and you must provide a list of the label names  
(i.e. the label to be predicted) and the protected attributes. In addition you can include
- title: Title for the dataset (default: Dataset_y-m-d_h:m:s)
- predictions: numpy array of predicted lables for the dataset
- categorical_features: Features of the dataset that are categories and should be numerized
- training_features: Features to be used for training
- model: A pre-trained model for the dataset
- map_func_(pa/lab/pred): If the protected attributes, labels and/or predictions are not numerical in the dataset you can pass simple convertion functions into the dataset object so that they are properly numerized  
- alter_dataframe: If dataframe is already in the correct format (mainly used by internal functions)

```python
import pandas as pd
from aibias.dataset import Dataset

df                   = pd.read_csv('path/to/dataset.csv')
label_names          = ['hired_status']
protected_attributes = ['age','gender']
title                = 'My Title'

dataset = Dataset(df=df,
                  label_names=label_names,
                  protected_attributes=protected_attributes,
                  title=title
                 )
```

#### Metrics

All metrics are implemented as simple functions and take in a Dataset object and the additional information if necessary.

```python
from aibias.metrics import DisparateImpact as DI

di = DI(dataset)
```

#### Algorithms

Currently there are three algorithms implemented, one pre-processing, one in-processing, and one post-processing.
They are all implemented as functions that take in a dataset along with additional information if necessary;
and they all return a new dataset object with the transformed dataset.

##### Pre-processing
The pre-processing algorithm implemented is called `Reweigh` and it adjust the weights of individuals so as to
make it such that unprivileged individuals with a positive label and privileged individuals with negative labels
weigh higher; while unprivileged individuals with a negative label and privileged individuals witha  positive labels
weigh lower.

```python
from aibias.algorithms.pre_processing import Reweigh
 
rw_dataset = Reweigh(dataset)
```

##### In-processing
The in-processing algorithm implemented is called `PrejudiceRemover` and it trains a simple linear regression model
that uses a special regularizer that punishes the model for relying on the protected attributes. In addition to the dataset
the function can also take in:
- epochs: Number of epochs to train the model (default: 1)
- eta: Scalar for the prejudice remover regularize (default: 0.5)
- ntrain: How many training samples to use from each category for protected attributes (e.g. 500 men and 500 women; selected randomly) (default: 1000)

Future releases will include greater control over the model by the user.  
Currently only supports one protected attribute at a time.

```python
from aibias.algorithms.in_processing import PrejudiceRemover

pr_dataset = PrejudiceRemover(dataset, epochs=10, eta=0.1, ntrain=500)
```

##### Post-processing
The post-processing algorithm is called `RejectOption` and takes in a dataset with predictions and makes it  
so that all predictions within the critical area for unprivileged individuals are positive, and for privileged  
individuals are negative. The critical are is defined as 0.5 ± Theta, where Theta is defined by the user.  
Note, does not work for discrete predictions.

```python
from aibias.algorithms.post_processing import RejectOption

ro_dataset = RejectOption(dataset,Theta=0.15)
```

#### Visualization

In order to create auto-generated visualizations, you must first create a Visualization object  
containing all the datasets to be visualized. Then that object can call visualzation functions.  
Currently two functions are implemented, `visualize_metric` which creates a graph displaying  
the given metric for each dataset, and `visualize_metrics` which calls the prior function for each  
implemented metric.  
For both you can also include the following:
- rotation: Rotation of dataset labels for graph (default: 90)
- annotation: To include the exact number on the graph (default: True)
- references: List of references for each dataset (i.e. whether to use labels or predictions)

They also support any keyword arguments that `matplotlib.pyplot.bar` supports.

```python
from aibias.visualization import Visualization as vis

datasets = [dataset,rw_dataset,pr_dataset,ro_dataset]
visual = vis(datasets)

vis.visualize_metric('DisparateImpact',rotation=45, color=green)
vis.visualize_metrics()
```

___

Made as a part of work at the Soft Computing Lab at Yonsei University, South-Korea
