# Maximum-Mean-and-Covariance-Discrepancy-Variational-AutoEncoder
Maximum Mean and Covariance Discrepancy Variational AutoEncoder (MMCD VAE)

### Details and motivation are described in the below paper.
https://thesai.org/Publications/ViewPaper?Volume=13&Issue=6&Code=IJACSA&SerialNo=104

## Usage

To use MMCD VAE, run `mmcd_vae.py` script.

## Dependencies
Python 3.x
Tensorflow 2.x
Keras

Output Comparison
Output comparison of KLVAE and MMCDVAE:

![image](https://user-images.githubusercontent.com/42261383/226728229-15b2b278-561f-47ee-b879-7cb0b62165c6.png)

## Dataset
The Bollywood dataset was used in this project. The dataset contains localized faces of Bollywood celebrities and is available on Kaggle:

https://www.kaggle.com/datasets/sushilyadav1998/bollywood-celeb-localized-face-dataset

## Credits
This project was inspired by the paper "Maximum Mean Discrepancy Variational Autoencoders" by S. Chang et al. (https://arxiv.org/pdf/1711.01558.pdf)

## Citation
If you use this code or the Bollywood dataset in your research, please cite the original paper:

@article{Barreto2022,
title = {Unsupervised Domain Adaptation using Maximum Mean Covariance Discrepancy and Variational Autoencoder},
journal = {International Journal of Advanced Computer Science and Applications},
doi = {10.14569/IJACSA.2022.01306104},
url = {http://dx.doi.org/10.14569/IJACSA.2022.01306104},
year = {2022},
publisher = {The Science and Information Organization},
volume = {13},
number = {6},
author = {Fabian Barreto and Jignesh Sarvaiya and Suprava Patnaik and Sushilkumar Yadav}
}
