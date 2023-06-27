# NLP Tools for Understanding the Role of Propaganda Techniques Online

Official PyTorch implementation of the Semester Project.

```pip install -r requirements```

## Data creation

The articles are in the ```data``` folder. To create csv files from data, run ```python data.py [parameters]```, where the parameters are:
```
    --taskType [OPTIONS: multi | single | binary, DEFAULT: multi]
    --labelsType [OPTIONS: propaganda | persuasion, DEFAULT: propaganda]
    --articlesDir [string]
    --fileOut [string]
    --title [OPTIONS: true | false, DEFAULT: false]
    --length [OPTIONS: true | false, DEFAULT: false]
    --position [OPTIONS: true | false, DEFAULT: false]
    --emotion [OPTIONS: true | false, DEFAULT: false]
```

The ```taskType``` parameter creates data for one of the chosen task: binary classification (option ```binary```), multilabel classification (option ```multi```), or single label classification (option ```single```). The ```labelsType``` parameter gets the 18 labels from propaganda (option ```propaganda```), or the 6 labels from persuasion (option ```persuasion```). Pass the articles directory with the ```articlesDir``` parameter and the path of the output file with the ```fileOut``` parameter. To add additional features, set the feature options (```title```, ```length```, ```position```, ```emotion```) to true. 

## Training

Create a ```models/``` folder if you want to save the weights of the model trained. Run ```python train.py [parameters]``` with the following parameters:
```
    --expID [string]
    --seed [int]
    --task [OPTIONS: multi | single | binary]
    --classType [string]
    --trainPath [string]
    --valPath [string]
    --saveModel [OPTIONS: true | false, DEFAULT: true]
    --savingPath [string]
    --model [OPTIONS: bert | roberta, DEFAULT: roberta]
    --nEpochs [int]
    --batchSize [int]
    --learningRate [float]
    --no-weightedTraing [NO OPTIONS]
    --weightedSampler [OPTIONS: true | false, DEFAULT: true]
    --labelsType [OPTIONS: persuasion | propaganda, DEFAULT: propaganda]
```

Set the ```expID```, ```classType``` to have a folder ```/models/classType/expID/``` on which you save the model (set the option ```saveModel``` to true) and the statistcs of training. ```model``` option choose the language model to use for the training. The ```weightedSampler``` set to true to use the weighted sampler. ```no-weightedTraining```, not to use the weights on the loss. 

## Test

Run ```python test.py [parameters]``` with parameters

```
    --binaryModel [string]
    --multilModel [string]
    --testPath [string]
```

Pass the path to the weights of the multilabel model (or also the binary if you want to test the pipeline) and the path to the data with ```testPath```.

