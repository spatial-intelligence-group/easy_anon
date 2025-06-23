#!/usr/bin/env python3


"""Anonymize the given images using pre-generated masks."""

import argparse
import os
import numpy as np
import cv2
from rich.progress import track
from rich.console import Console

from easy_anon.utils import check_img_ext, check_mask_ext, IMG_EXTS, MASK_EXTS


def main():
    """Anonymize the given images using pre-generated masks."""
    parser = argparse.ArgumentParser(
        description="Masking / anonymization using Mask2Former",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to the input image / image directory",
    )
    parser.add_argument(
        "--input_mask",
        type=str,
        required=True,
        help="Path to the input mask / mask directory."
        "If the input is a directory, the masks are associated to the images based on their basenames."
        "The basenames are compared without file extensions and with 'mask_postfix' removed from the mask basename.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output anonymized image / directory for anonymized images",
    )
    parser.add_argument(
        "--infill_mode",
        type=str,
        default="average_border",
        choices=["average_inside", "average_border", "single_color", "blur_box", "blur_gauss", "inpaint"],
        help="The infill mode to use for anonymization. "
        "Options are:"
        "'average_inside': Fill with the average color of each enclosed masked region. "
        "'average_border': Fill with the average color of the border just outside the masked region. "
        "'single_color': Fill with a single color (can be specified with '--single_color'). "
        "'blur_box': Fill with a blurred version of the masked region using box blur. "
        "'blur_gauss': Fill with a blurred version of the masked region using Gaussian blur. "
        "'inpaint': Fill with inpainting.",
    )
    parser.add_argument(
        "--mask_color_mode",
        type=str,
        default="black_on_white",
        choices=["black_on_white", "white_on_black"],
        help="Color mode of the given masks. "
        "If 'black_on_white', the black regions will be anonymized. "
        "If 'white_on_black', the white regions will be anonymized.",
    )
    parser.add_argument(
        "--mask_postfix",
        type=str,
        default="",
        help="The postfix of the mask file names, which is remove before comparing the basenames with images.",
    )
    parser.add_argument(
        "--anon_postfix",
        type=str,
        default="",
        help="The postfix added to the anonymized image files. "
        "Can be used to set a specific output image format by adding an extension to the postfix. "
        "If not given, the anonymized image will have the same name as the input image (including image format).",
    )
    parser.add_argument(
        "--single_color",
        type=int,
        nargs="+",
        default=[0, 0, 0],
        help="The RGB color to use when infill mode is 'single_color'. "
        "Format: 8-bit integer RGB or RGBA values (e.g. '255 0 0' for red).",
    )
    parser.add_argument(
        "--size_param",
        type=float,
        default=None,
        help="Value used as the kernel size for blurring and as the neighborhood radius for inpainting. "
        "If None, it is set to 1/10 of the longer image side length for blurring and to 20 pixels for inpainting. "
        "If set to a value >= 1, it is considered to be a value in pixels. "
        "If set to a vlues > 0 and < 1, it is interpreted as a fraction of the longer image side length.",
    )
    parser.add_argument(
        "--delete_mask",
        action="store_true",
        help="Delete the binary masks after anonymization",
    )
    args = parser.parse_args()
    console = Console()

    assert len(args.single_color) in [3, 4], "The single_color argument must contain 3 or 4 values (RGB or RGBA)."

    args.mask_postfix = os.path.splitext(args.mask_postfix)[0]  # Remove file extension if given

    img_is_file = os.path.isfile(args.input_image)
    mask_is_file = os.path.isfile(args.input_mask)
    img_is_dir = os.path.isdir(args.input_image)
    mask_is_dir = os.path.isdir(args.input_mask)

    if not (img_is_file or img_is_dir):
        raise ValueError(f"The given input image path does not exist: {args.input_image}")
    if not (mask_is_file or mask_is_dir):
        raise ValueError(f"The given input mask path does not exist: {args.input_mask}")

    if img_is_file:
        if not check_img_ext(args.input_image):
            raise ValueError(
                f"The given input image is not a valid image file: {args.input_image} "
                f"Supported image extensions are: {', '.join(IMG_EXTS)}"
            )
        input_image_list = [args.input_image]
    elif img_is_dir:
        input_image_list = [os.path.join(args.input_image, f) for f in os.listdir(args.input_image) if check_img_ext(f)]
        if not input_image_list:
            raise ValueError(
                f"The given input image directory does not contain any valid image files: {args.input_image} "
                f"Supported image extensions are: {', '.join(IMG_EXTS)}"
            )

    if mask_is_file:
        if not check_mask_ext(args.input_mask):
            raise ValueError(
                f"The given input mask is not a valid mask file: {args.input_mask} "
                f"Supported mask extensions are: {', '.join(MASK_EXTS)}"
            )
        input_mask_list = [args.input_mask]
    elif mask_is_dir:
        input_mask_list = [os.path.join(args.input_mask, f) for f in os.listdir(args.input_mask) if check_mask_ext(f)]
        if not input_mask_list:
            raise ValueError(
                f"The given input mask directory does not contain any valid mask files: {args.input_mask} "
                f"Supported mask extensions are: {', '.join(MASK_EXTS)}"
            )

    input_image_basenames = [os.path.splitext(os.path.basename(f))[0] for f in input_image_list]
    input_mask_basenames = [
        os.path.splitext(os.path.basename(f))[0].removesuffix(args.mask_postfix) for f in input_mask_list
    ]

    unpaired_images = [
        os.path.basename(input_image_list[idx])
        for idx in range(len(input_image_basenames))
        if input_image_basenames[idx] not in input_mask_basenames
    ]
    unpaired_masks = [
        os.path.basename(input_mask_list[idx])
        for idx in range(len(input_mask_basenames))
        if input_mask_basenames[idx] not in input_image_basenames
    ]
    paired_basenames = [img for img in input_image_basenames if img in input_mask_basenames]

    if unpaired_images:
        console.print(
            "Warning :warning: : The following input images do not have corresponding masks: \n- "
            + "\n- ".join(unpaired_images),
            style="yellow",
        )
    if unpaired_masks:
        console.print(
            "Warning :warning: : The following input masks do not have corresponding images: \n- "
            + "\n- ".join(unpaired_masks),
            style="yellow",
        )

    console.print(f"Found {len(paired_basenames)} image-mask pairs for anonymization")
    for file_basename in track(paired_basenames, description="Anonymizing..."):
        img_idx = input_image_basenames.index(file_basename)
        mask_idx = input_mask_basenames.index(file_basename)

        img_path = input_image_list[img_idx]
        mask_path = input_mask_list[mask_idx]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        if img is None:
            raise ValueError(f"Could not read the image file: {img_path}")
        mask = load_mask(mask_path, mode=args.mask_color_mode)

        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"The image and mask sizes do not match: {img_path} ({img.shape}) vs {mask_path} ({mask.shape})"
            )

        # Anonymize the image using the mask
        img_anon = anonymize_image(
            img,
            mask,
            mode=args.infill_mode,
            single_color=args.single_color,
            size_param=args.size_param,
        )

        # Save the anonymized image
        output_filename = os.path.basename(img_path)
        if args.anon_postfix:
            output_filename = os.path.splitext(output_filename)[0] + args.anon_postfix
            if not os.path.splitext(args.anon_postfix)[1]:
                output_filename += os.path.splitext(img_path)[1]
        output_path = os.path.join(args.output, output_filename)

        img_anon = cv2.cvtColor(img_anon, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_anon)


def load_mask(mask_path, mode="black_on_white"):
    """Load the mask and convert it to the desired color mode.

    Args:
        mask_path (str): Path to the mask file.
        mode (str): Color mode of the mask. Options are:
            - "black_on_white": Black regions will be anonymized.
            - "white_on_black": White regions will be anonymized.

    Returns:
        np.ndarray: The binary mask where the regions to be anonymized are set to True.
    """
    if os.path.splitext(mask_path)[1].lower() == ".npy":
        mask = np.load(mask_path)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    elif os.path.splitext(mask_path)[1].lower() == ".npz":
        mask = np.load(mask_path)["mask"]
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Could not read the mask file: {mask_path}")

    if mask.dtype == np.uint8:
        mask = mask.astype(np.float32)
        mask = mask / 255.0

    # 0 and 1 should be safe for direct comparison
    if mode == "black_on_white":
        mask = mask == 0
    elif mode == "white_on_black":
        mask = mask == 1

    return mask


def anonymize_image(img, mask, mode="average_color", single_color=[0, 0, 0], size_param=None):
    """Anonymize the image in the regions defined by the mask.

    Args:
        img (np.ndarray): The input image to be anonymized.
        mask (np.ndarray): The binary mask defining the regions to be anonymized.
        mode (str): The infill mode to use for anonymization. Options are:
            - "average_inside": Fill with the average color of each enclosed masked region.
            - "average_border": Fill with the average color of border of each enclosed masked region.
            - "single_color": Fill with a single color (black).
            - "blur_box": Fill with a blurred version of the masked region using box blur.
            - "blur_gauss": Fill with a blurred version of the masked region using Gaussian blur
            - "inpaint": Fill with inpainting.
        single_color (list): The RGB or RGBA color (list of length 3 or 4 of 8-bit integers) for "single_color" mode.
        size_param (int): The kernel size for blurring or the neighborhood radius for inpainting.
            If None, it is computed as 1/10 of the longer image side length.

    Returns:
        np.ndarray: The anonymized image.
    """
    img_anon = img.copy()

    if mode == "average_inside":
        img_anon = anonymize_average_inside(img, mask)
    if mode == "average_border":
        img_anon = anonymize_average_border(img, mask)
    elif mode == "single_color":
        img_anon = anonymize_single_color(img, mask, single_color)
    elif mode in ["blur_box", "blur_gauss"]:
        img_anon = anonymize_blur(img, mask, blur_type=mode, blur_kernel_size=size_param)
    elif mode == "inpaint":
        img_anon = anonymize_inpaint(img, mask, neighborhood_radius=size_param)
    else:
        raise ValueError(f"Unknown infill mode: {mode}")

    return img_anon


def anonymize_average_inside(img, mask):
    """Anonymize the image using the average color in each masked region.

    Args:
        img (np.ndarray): The input image to be anonymized.
        mask (np.ndarray): The binary mask defining the regions to be anonymized.

    Returns:
        np.ndarray: The anonymized image with the average color filled in the masked regions.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        avg_color = cv2.mean(img, mask=contour_mask)[:3]
        img[contour_mask.astype(bool)] = avg_color

    return img


def anonymize_average_border(img, mask):
    """Anonymize the image using the average color just outside the masked regions.

    Args:
        img (np.ndarray): The input image to be anonymized.
        mask (np.ndarray): The binary mask defining the regions to be anonymized.

    Returns:
        np.ndarray: The anonymized image with the average color of the border filled in the masked regions.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        contour_mask_dilated = cv2.dilate(contour_mask, np.ones((3, 3), np.uint8), iterations=1)
        contour_mask_border = cv2.bitwise_and(contour_mask_dilated, cv2.bitwise_not(contour_mask))
        avg_color = cv2.mean(img, mask=contour_mask_border)[:3]
        img[contour_mask.astype(bool)] = avg_color

    return img


def anonymize_single_color(img, mask, single_color):
    """Anonymize the image using a single color in the masked regions.

    Args:
        img (np.ndarray): The input image to be anonymized.
        mask (np.ndarray): The binary mask defining the regions to be anonymized.
        single_color (list): The RGB or RGBA color (list of length 3 or 4 of 8-bit integers) for infill.

    Returns:
        np.ndarray: The anonymized image with the single color filled in the masked regions.
    """
    img_anon = img.copy()

    if len(single_color) == 3:
        img_anon[mask] = single_color
    elif len(single_color) == 4:
        # If RGBA, the infill will be blended with the original image based on the alpha channel
        alpha = single_color[3] / 255.0
        single_color = single_color[:3]
        img_anon[mask] = img[mask] * (1 - alpha) + np.array(single_color[:3]) * alpha

    return img_anon


def anonymize_blur(img, mask, blur_type, blur_kernel_size=None):
    """Anonymize the image using a blur in the masked regions.

    Args:
        img (np.ndarray): The input image to be anonymized.
        mask (np.ndarray): The binary mask defining the regions to be anonymized.
        blur_type (str): The type of blur to use ("blur_box" or "blur_gauss").
        blur_kernel_size (int): The kernel size for blurring.
            If None, it is computed as 1/10 of the longer image side length.

    Returns:
        np.ndarray: The anonymized image with the blurred regions.
    """
    if blur_kernel_size is None:
        blur_kernel_size = max(img.shape[:2]) // 10
    elif blur_kernel_size < 1:
        blur_kernel_size = int(max(img.shape[:2]) * blur_kernel_size)

    if blur_type == "blur_box":
        img_blur = cv2.blur(img, (blur_kernel_size, blur_kernel_size))
    elif blur_type == "blur_gauss":
        blur_kernel_size = blur_kernel_size | 1  # Ensure the kernel size is odd
        img_blur = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
    else:
        raise ValueError(f"Unknown blur type: {blur_type}")

    img_anon = img.copy()
    img_anon[mask] = img_blur[mask]

    return img_anon


def anonymize_inpaint(img, mask, neighborhood_radius=20):
    """Anonymize the image using inpainting in the masked regions.

    Args:
        img (np.ndarray): The input image to be anonymized.
        mask (np.ndarray): The binary mask defining the regions to be anonymized.
        neighborhood_radius (int): The radius of sampling neighborhood used for inpainting.

    Returns:
        np.ndarray: The anonymized image with the inpainted regions.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    img_anon = cv2.inpaint(img, mask_uint8, inpaintRadius=neighborhood_radius, flags=cv2.INPAINT_TELEA)

    return img_anon


if __name__ == "__main__":
    main()
