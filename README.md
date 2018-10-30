# Probabilistic Semantic Inpainting with Pixel Constrained CNNs

Pytorch implementation of [Probabilistic Semantic Inpainting with Pixel Constrained CNNs](https://arxiv.org/abs/1804.00104) (2018).

This repo contains all code to reproduce the experiments in the paper as well as all the trained model weights.

## Examples

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/summary-figure.png" width="400">

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/grid-progression-celeba-row.gif" width="500">

#### Samples sorted by their likelihood

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/eye-completions-likelihood.png" width="400">

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/mnist-likelihood.png" width="400">

#### Pixel probabilities during sampling

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/logit-progression-1.gif" width="200">

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/logit-progression-2.gif" width="200">

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/logit_1_from_1.png" width="400">

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/logit_3_from_3.png" width="400">

#### Architecture

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/architecture.png" width="400">

## Usage

To train a model, run (ensure that you have either the CelebA or MNIST dataset downloaded)

```
python main.py config.json
```

To generate images with a trained model use `main_generate.py`. As an example, the following command generates 64 completions for images 73 and 84 in the MNIST dataset by conditioning on the bottom 7 rows. The model used to generate the completions is the trained MNIST model included in this repo and the results are saved to the `mnist_experiment` folder.

```
python main_generate.py -n mnist_experiment -m trained_models/mnist -t generation -i 73 84 -b 7 -ns 64
```

## Trained models

The trained models referenced in the paper are included in the `trained_models` folder. You can use the `main_generate.py` script to generate image completions (and other plots) with these models.

## Data sources

The MNIST dataset can be automatically downloaded using `torchvision`. All CelebA images were resized to be 32 by 32. Data can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Citing

If you find this work useful in your research, please cite using:

```
@article{dupont2018probabilistic,
  title={Probabilistic Semantic Inpainting with Pixel Constrained CNNs},
  author={Dupont, Emilien and Suresha, Suhas},
  journal={arXiv preprint arXiv:1810.03728},
  year={2018}
}
```

## More examples

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/blob-samples-celeba.png" width="400">

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/blob-samples-celeba-2.png" width="400">

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/blob-samples-mnist.png" width="400">

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/bottom-samples-celeba.png" width="400">

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/bottom-samples-celeba-2.png" width="400">

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/logit_6_from_6.png" width="400">

<img src="https://github.com/Schlumberger/pixel-constrained-cnn/raw/master/open-source/imgs/logit_5_from_0.png" width="400">

## License

MIT
