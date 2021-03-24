
# Imbalanced CCSA Loss

This repository contains:

* code for the dynamic sampler and weighted alignment loss

* instructions for downloading the 7 skin datasets

* a toy dataset as placeholder data to demonstrate an end-to-end pipeline

## Files

### Dynamic Sampler

An implementation of the dynamic sampler is in: `datasets/toy.py`. Note that when using multiple workers, it is important to seed each worker independently as otherwise each worker willuse the same random seed:
```
def  worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)

loader = torch.utils.data.DataLoader(
	dataset,
	num_workers=self.config.num_workers,
	worker_init_fn=worker_init_fn
)
```
### Weighted Alignment Loss

The weighted alignment loss is implemented in `graphs/loss.py` and an example usage can be found in `agents/toy.py`
  

## Running experiments

Visdom is used to track the learning process. Start visdom before running the main script. The main script is called by:
```
python main.py configs/base.hocon
```

This will generate a synthetic toy dataset and run a simple example

  

## Downloading the datasets used

### HAM10000

The HAM10000 could be downloaded from multiple sources such as:

* https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

* https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

  

### MSK, UDA, SONIC

The three datasets (MSK, UDA, SONIC) were downloaded from the ISIC archive:

* https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main

Files can be downloaded using the ISIC API directly: https://isic-archive.com/api/v1/ or using ISIC Archive Download project from: https://github.com/GalAvineri/ISIC-Archive-Downloader

  

### Dermofit

The Dermofit dataset can be obtained from the University of Edinburgh:

* https://licensing.edinburgh-innovations.ed.ac.uk/i/software/dermofit-image-library.html

  

### Derm7pt

The 7-point Criteria Evaluation Database is available from Simon Fraser University:

* https://derm.cs.sfu.ca/Welcome.html

  

### PH2

The PH2 dataset can be downloaded from:

* https://www.fc.up.pt/addi/ph2%20database.html


## Citation
This is the code corresponding to our MICCAI 2019 paper, 

*Chris Yoon, Ghassan Hamarneh, and Rafeef Garbi*. [Generalizable Feature Learning in the Presence of Data Bias and Domain Class Imbalance with Application to Skin Lesion Classification](https://www.springerprofessional.de/en/generalizable-feature-learning-in-the-presence-of-data-bias-and-/17255502). In *Lecture Notes in Computer Science, Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, volume 11767, pages 365-373, 2019.

```
@InProceedings{Yoon_2019_MICCAI,
author = {Yoon, Chris and Hamarneh, Ghassan and Garbi, Rafeef},
title = {Generalizable Feature Learning in the Presence of Data Bias and Domain Class Imbalance with Application to Skin Lesion Classification},
booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
month = {October},
year = {2019}
}
```
