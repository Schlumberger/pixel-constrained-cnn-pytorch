# Probabilistic Semantic Inpainting with Pixel Constrained CNNs

Pytorch implementation of [Probabilistic Semantic Inpainting with Pixel Constrained CNNs](https://arxiv.org/abs/1810.03728) (2018).

This repo contains an implementation of Pixel Constrained CNN, a framework for performing probabilistic inpainting of images with arbitrary occlusions. It also includes all code to reproduce the experiments in the paper as well as the weights of the trained models.

For a TensorFlow implementation, see this [repo](https://github.com/Schlumberger/pixel-constrained-cnn-tf).

## Examples

<img src="https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/raw/master/imgs/summary-gif.gif" width="600">
<img src="https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/raw/master/imgs/probinpainting.gif" width="400">
<img src="https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/raw/master/imgs/new-top-celeba.png" width="400">
<img src="https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/raw/master/imgs/new-left-celeba.png" width="400">
<img src="https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/raw/master/imgs/new-small-missing-celeba.png" width="400">

<img src="https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/raw/master/imgs/new-blob-samples.png" width="400">
<img src="https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/raw/master/imgs/new-bottom-seven-nine.png" width="400">

<img src="https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/raw/master/imgs/int-dark-side.png" width="300">
<img src="https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/raw/master/imgs/int-eye-color.png" width="300">

## Usage

### Training

The attributes of the model can be set in the `config.json` file. To train the model, run

```
python main.py config.json
```

This will also save the trained model and log various information as training progresses. Examples of `config.json` files are available in the `trained_models` directory.

### Inpainting

To generate images with a trained model use `main_generate.py`. As an example, the following command generates 64 completions for images 73 and 84 in the MNIST dataset by conditioning on the top 14 rows. The model used to generate the completions is the trained MNIST model included in this repo and the results are saved to the `mnist_inpaintings` folder.

```
python main_generate.py -n mnist_inpaintings -m trained_models/mnist -t generation -i 73 84 -to 14 -ns 64
```

For a full list of options, run `python main_generate.py --help`. Note that if you do not have the MNIST dataset on your machine it will be automatically downloaded when running the above command. The CelebA dataset will have to be manually downloaded (see the Data sources section). If you already have the datasets downloaded, you can change the paths in `utils/dataloaders.py` to point to the correct folders on your machine.

## Trained models

The trained models referenced in the paper are included in the `trained_models` folder. You can use the `main_generate.py` script to generate image completions (and other plots) with these models.

## Data sources

The MNIST dataset can be automatically downloaded using `torchvision`. The CelebA dataset can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

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

<img src="https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/raw/master/imgs/new-random-celeba.png" width="400">
<img src="https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/raw/master/imgs/new-bottom-celeba.png" width="400">
<img src="https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/raw/master/imgs/new-blob-samples-2.png" width="400">
<img src="https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/raw/master/imgs/int-male-female.png" width="300">

## License

[Apache License 2.0](LICENSE)
