# Probabilistic Semantic Inpainting with Pixel Constrained CNNs

Pytorch implementation of [Probabilistic Semantic Inpainting with Pixel Constrained CNNs](https://arxiv.org/abs/1804.00104) (2018).

This repo contains an implementation of Pixel Constrained CNN, a framework for performing probabilistic inpainting of images with arbitrary occlusions. It also includes all code to reproduce the experiments in the paper as well as the weights of the trained models.

## Examples

<img src="https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/raw/master/imgs/summary-gif.gif" width="600">
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

This will also save the trained model and log various information as training progresses.

### Inpainting

To generate images with a trained model use `main_generate.py`. As an example, the following command generates 64 completions for images 73 and 84 in the CelebA dataset by conditioning on the top 16 rows. The model used to generate the completions is the trained CelebA model included in this repo and the results are saved to the `celeba_experiment` folder.

```
python main_generate.py -n celeba_experiment -m trained_models/celeba -t generation -i 73 84 -to 16 -ns 64
```

For a full list of options, run `python main_generate.py --help`.

## Trained models

The trained models referenced in the paper are included in the `trained_models` folder. You can use the `main_generate.py` script to generate image completions (and other plots) with these models.

## Data sources

The MNIST dataset can be automatically downloaded using `torchvision`. All CelebA images were resized to 32 by 32 and can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

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

MIT
