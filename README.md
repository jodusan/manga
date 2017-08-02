# Project MANGA PSIML2017

This repo contains project that we created during the [Petnica Summer Institue: Machine Learning](http://psiml.petlja.org/) seminar. For intro to the project I suggest that you skim through presentation slides on the master branch in the misc folder.

# Datasets

Dataset images can be _loaded_ or *patched* from a giant montage of all images.

-   `Manga546` -  546 images of color and bw - size 950x640

    Created out of ScottPilgrim comics and is not openly available, but it we weren't able to make much use of it.

-   `MangaOnline` -  42215 images of color - size 512x512

    Consists of anime/manga images and can be downloaded from the safebooru.org website using the script that is created by @kvfrans and provided in the master branch.

# Ideas

Every list entry is an idea implemented on the branch of the same name

-   `UNET`  -  U-net autoencoder on *patched* **Manga546**
-   `UNET-LOAD` - U-net autoencoder on _loaded_ **Manga546** (whole images are loaded)
-   `UNET-ONLINE` - U-net autoencoder on loaded **MangaOnline**
-   `UNET-ONLINE-HINT` - U-net autoencoder trained with image hints on _loaded_ **MangaOnline**
-   `simple-segmentated` - Few convolutional layers on _patched_ **Manga546**
-   `simple-conv` -  Few convolutional layers on _patched_ **Manga546**



## Installation

```sh
pip3 install opencv-python tensorflow keras h5py scikit-image
git clone https://github.com/dulex123/manga

# This is our best model, you can use any model/branch
git checkout UNET-ONLINE-HINT
```

## Usage

**Only valid README is on the master branch**

Datasets should be downloaded at the following locations:

-   `data/MangaOnline/`
-   `data/Manga546/`

In order to run prediction on a single image you can use predict.py files while the training can be executed by running the branch-name.py file.

```python
python3 predict.py img/to/predict.ext
```

# Credits

Released under the [MIT License].<br>
Authored and maintained by  Dušan Josipović & Nikola Jovičić.

> Github [@nikola-j](https://github.com/nikola-j) &nbsp;&middot;&nbsp;

> Blog [dulex123.github.io](http://dulex123.github.io) &nbsp;&middot;&nbsp;
> GitHub [@dulex123](https://github.com/dulex123) &nbsp;&middot;&nbsp;
> Twitter [@josipovicd](https://twitter.com/josipovicd)

[MIT License]: http://mit-license.org/
