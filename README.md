# A Novel Contrastive Loss for Zero-Day Network Intrusion Detection
<p align="center">
  <img src="images/architecture.png" width="700">
</p>

Official Github repo for the paper "A Novel Contrastive Loss for Zero-Day Network Intrusion Detection". This repository covers a reference implementation for both the Contrastive Learning for Anomaly Detection (CLAD) and Contrastive Learning for Open Set Recognition (CLOSR) methods decribed in [this work](www.google.com).

## CLAD

The CLAD loss function models benign traffic as a vMF distribution in embedded space. The loss function [`CLADLoss`](https://github.com/jackwilkie/CLOSR/blob/main/losses/clad_loss.py#L75) in `losses/clad_loss.py` takes `x` (L2 normalised features) and `y` (labels) as inputs, and returns the loss.

Usage:

```python
from losses.clad_loss import CLADLoss

# define CLAD loss with a margin `m`
criterion = CLADLoss(m = m)

# features: [bsz, f_dim]
# features should L2 normalized in f_dim dimension
features = ...
# labels: [bsz]
labels = ...

# calculate loss
loss = criterion(x = features, y = labels)
...
```


## CLOSR

The CLOSR loss function models each class as a vMF distribution a distinct embedded space. The loss function [`CLSORLoss`](https://github.com/jackwilkie/CLOSR/blob/main/losses/closr_loss.py#L52) in `losses/closr_loss.py` takes `x` (L2 normalised features for each class) and `y` (labels) as inputs, and returns the loss.

```python
from losses.closr_loss import CLOSRLoss

# define CLOSR loss with a margin `m` and 'n_classes' classes
criterion = CLADLoss(m = m, n_classes = n_classes)

# features: [bsz, n_classes, f_dim]
# features should L2 normalized in f_dim dimension
features = ...
# labels: [bsz]
labels = ...

# calculate loss
loss = criterion(x = features, y =labels)
...
```

## Comparison
CLAD significantly outperforms both closed-set classifiers on known attack detection (top) and anomaly detectors on zero-day attack detection (bottom):

| Class                    | CLAD     | DUAD     | DAE-LR   | Deep SVDD | AE       | SVM      | AutoSVM  | IF       | RENOIR   | MLP      | Siamese  |
|--------------------------|----------|----------|----------|-----------|----------|----------|----------|----------|----------|----------|----------|
| Botnet                   | 0.999980 | 0.888620 | 0.720118 | 0.626321  | 0.648185 | 0.637679 | 0.662214 | 0.696588 | 0.999990 | 0.999982 | 0.999992 |
| DDoS                     | 0.999995 | 0.987811 | 0.998323 | 0.971817  | 0.957563 | 0.889780 | 0.722085 | 0.778166 | 0.999996 | 0.999995 | 0.999881 |
| DoS (Golden Eye)         | 0.999585 | 0.930244 | 0.954869 | 0.740828  | 0.878592 | 0.846840 | 0.796121 | 0.727087 | 0.999425 | 0.999397 | 0.998539 |
| DoS (Hulk)               | 0.999995 | 0.988089 | 0.993563 | 0.899340  | 0.950815 | 0.894898 | 0.793412 | 0.781731 | 0.999991 | 0.999985 | 0.999778 |
| DoS (Slow HTTP Test)     | 0.999575 | 0.911725 | 0.975763 | 0.659927  | 0.978112 | 0.963021 | 0.925308 | 0.848820 | 0.999831 | 0.999543 | 0.999474 |
| DoS (Slow Loris)         | 0.999876 | 0.947156 | 0.980669 | 0.692552  | 0.938116 | 0.896824 | 0.776275 | 0.775622 | 0.999928 | 0.999937 | 0.999826 |
| FTP Patator              | 0.999986 | 0.982829 | 0.952138 | 0.822762  | 0.779748 | 0.736828 | 0.702073 | 0.759742 | 0.999991 | 0.999987 | 0.999948 |
| Portscan                 | 0.999993 | 0.943170 | 0.974972 | 0.732919  | 0.871651 | 0.741269 | 0.584936 | 0.742066 | 0.999986 | 0.999962 | 0.999823 |
| SSH Patator              | 0.999930 | 0.961050 | 0.961583 | 0.671834  | 0.824424 | 0.799072 | 0.789865 | 0.816086 | 0.999951 | 0.999980 | 0.999978 |
| Web Attack (Brute Force) | 0.999754 | 0.836997 | 0.731959 | 0.635410  | 0.801017 | 0.767771 | 0.744257 | 0.748669 | 0.999312 | 0.998011 | 0.996850 |
| Web Attack (XSS)         | 0.999732 | 0.909319 | 0.773056 | 0.640942  | 0.801203 | 0.761677 | 0.734927 | 0.768263 | 0.999286 | 0.998353 | 0.996466 |
| **Closed Set Mean**      | **0.999855*** | 0.935183 | 0.910637 | 0.735877  | 0.857221 | 0.812333 | 0.748316 | 0.767531 | 0.999790 | 0.999557 | 0.999141 |
| Heartbleed               | 0.995557 | 0.987390 | 0.999798 | 0.985815  | 0.995161 | 0.993468 | 0.988778 | 0.955030 | 0.780736 | 0.071957 | 0.692232 |
| Web Attack (SQL Injection)| 0.997696 | 0.884098 | 0.777752 | 0.717808  | 0.686157 | 0.745087 | 0.767333 | 0.721814 | 0.991721 | 0.995762 | 0.997427 |
| **Open Set Mean**        | **0.996627*** | 0.935744 | 0.888775 | 0.851812  | 0.840659 | 0.869277 | 0.767333 | 0.838422 | 0.886228 | 0.533859 | 0.844829 |


CLOSR significantly improves open set recognition performance at a slight cost to closed set classificaiton performance when compared to baseline models:

| Model                  | Closed Set Acc | Open Set AUC       | OpenAUC            |
|-------------------------|----------------|--------------------|--------------------|
| CLOSR                  | 0.995276       | **0.974022***      | **0.969420***      |
| MultiStage             | 0.996612       | 0.801251           | 0.798537           |
| DOC                    | 0.995536       | 0.570263           | 0.567717           |
| OPENMAX                | 0.995615       | 0.720174           | 0.717016           |
| CRSOR                  | 0.994940       | 0.748295           | 0.744509           |
| Siamese Network        | **0.997722***  | 0.720811           | 0.719167           |

## Running 

### (1) Install Requirements

This repository requires python3 and Pytorch. To install the required dependencies run:

```
pip install -r requirements.txt
```

### (2) Download Dataset

Models and baselines are trained and evaluated on the **Lycos2017** dataset.

You can either:
- Download the original CSV files from [here](https://lycos-ids.univ-lemans.fr), combine them into a single file, and name it `lycos.csv`,  
**or**
- Download a preprocessed version directly from [here](https://drive.google.com/file/d/1PUMAbjz5L0MKiL3P-bEFTYYOcB6zPYB2/view?usp=share_link).

Once downloaded, move the file to: `./data/lycos.csv`.

### (3) Train Models

CLAD can be trained using the [train_clad.py](./train_clad.py) script:

```
python3 train_clad.py
```

CLOSR can be trained using the [train_closr.py](./train_closr.py) script:

```
python3 train_closr.py
```

Both scripts will train the model and save the weights to `./weights/clad.pt.tar` and `./weights/closr.pt.tar` for CLAD and CLOSR, respectively.

### (4) Evaluate Models

CLAD can be evaluated by running [eval_clad.py](./eval_clad.py):

```
python3 eval_clad.py
```

CLOSR can be evaluated by running [eval_closr.py](./eval_closr.py):

```
python3 eval_closr.py
```

The performance metrics of each model will be printed to the terminal after evaluation.

## t-SNE Visualisation

**(1) Contrastive Loss**
<p align="center">
  <img src="images/contrastive_tsne.png" width="400">
</p>

**(2) CLAD Loss**
<p align="center">
  <img src="images/clad_tsne.png" width="400">
</p>

## Reference
```
@Article{wilkie2025clad,
    title   = {A Novel Contrastive Loss for Zero-Day Network Intrusion Detection},
    author  = {Jack Wilkie and Hanan Hindy and Craig Michie and Christos Tachtatzis and James Irvine and Robert Atkinson},
    year    = {2025},
}
```