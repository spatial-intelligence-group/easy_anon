# An easy-to-use image masking and anonymization tool

What the tool does:
- Generates masks using semantic segmentation (e.g., for people, vehicles, sky, etc.).
- Can use the masks for anonymization.

In our experience the available open anonymization tools (for faces, license plates, ...) are not very robust on real-world data and require a lot of manual interventions. This tool:
1. Masks entire people and vehicles (larger objects easier to segment).
2. Uses a union of masks from multiple Mask2Former models (to limit failures of individual models).


## Installation

Torch, NumPy and Hatchling are needed at build time. You can either install them manually and then install the `easy_anon` package, which defines the runtime dependencies:
```bash
pip install torch numpy hatchling
pip install git+... --no-build-isolation
```
or you can use the complete (tested) environment defined in `requirements.txt` and then install just the `easy_anon` package:
```bash
pip install -r requirements.txt
pip install git+... --no-build-isolation --no-deps
```

- :green_heart: Mask2Former is installed automatically as a dependency.
- :warning: Mask2Former needs CUDA installed on the system.
    - If your installation of CUDA toolkit is not in `/usr/local/cuda`, you need to set the environment variable `CUDA_HOME` before installing Mask2Former.
- :bulb: We recommend using [uv](https://docs.astral.sh/uv/) and replacing the `pip` calls with `uv pip`. It's much faster!


## Usage

The tool does two separate things:
1. Generates masks using Mask2Former models.
2. Anonymizes images using the generated masks.

The most basic way to run the two is:
```bash
python -m easy_anon.mask <input_image_dir> <mask_dir>
python -m easy_anon.anon <input_image_dir> <mask_dir> <anonymized_images_dir>
```


### Generating masks

|  |  |
|:-----------------------:|:-----------------------:|
| ![input image](docs/images/image.jpg) | ![generated mask](docs/images/mask.png) |
| **input image (by [Anton Bielousov](https://commons.wikimedia.org/wiki/File:Muggle_Quidditch_Game_in_Vancouver.jpg))** | **generated mask** |

The parts of the image that are masked can be specified using the `--labels` argument. Current options contain labels for people (`person`), vehicles (`vehicle`), sky (`sky`), and more. These are groups of segmentation IDs of the individual Mask2Former models, which are defined in the labels config files in [`src/easy_anon/configs`](src/easy_anon/configs). Multiple label groups can be used at once (just specify list of the label groups `--labels person vehicle`). New label groups can be specified by changing the config files.

Mask2Former models used for the segmentation can be specified using the `--model` argument. Multiple models can be used at once (just specify list of the model names `--model ADE20k-ResNet101 ADE20k-Swin-L-IN21k`). By default the union of individual masks (from different models) is used to generate the final mask. The models should be automatically downloaded when first used.

Use help to get all the available options:
``` bash
python -m easy_anon.mask --help
```


### Anonymizing images

The way how the masked areas are filled in can be specified with the `--infill_mode` argument. The available options are:

|  |  |
|:-----------------------:|:-----------------------:|
| ![average_inside](docs/images/anon_average_inside.jpg) | ![average_border](docs/images/anon_average_border.jpg) |
| **average color in the masked area**<br>`average_inside` | **average color on the mask border**<br>`average_border` |
| ![single_color](docs/images/anon_single_color.jpg) | ![inpaint](docs/images/anon_inpaint.jpg) |
| **single specified color**<br>`single_color` | **inpainted by [\[Telea, 2004\]](https://doi.org/10.1080/10867651.2004.10487596)**<br>`inpaint` |
| ![blur_box](docs/images/anon_blur_box.jpg) | ![blur_gauss](docs/images/anon_blur_gauss.jpg) |
| **blurred with box filter**<br>`blur_box` | **blurred with Gaussian filter**<br>`blur_gauss` |

The color for the `single_color` infill mode can be specified using the `--single_color` argument (supports alpha). The blurring and inpainting can be adjusted with the `--size_param` argument.

Use help to get all the available options:
``` bash
python -m easy_anon.anon --help
```


## License
This project is licensed under MIT License - see the [LICENSE](LICENSE) file for details. Also check the [NOTICE](NOTICE) file for additional information.


## Acknowledgements
This project uses:
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) for semantic segmentation
- [OpenCV](https://opencv.org/) for image processing
- [Rich](https://github.com/Textualize/rich) for CLI
- and otters