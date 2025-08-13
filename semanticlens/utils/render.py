"""
Image visualization utilities for semantic analysis and heatmap rendering.
"""

import numpy as np
import torch
from crp.image import get_crop_range, imgify
from PIL import Image, ImageDraw, ImageFilter
from torchvision.transforms.functional import gaussian_blur
from zennit.core import stabilize


def _get_square_crop_box(heatmap: torch.Tensor, crop_th: float) -> tuple[int, int, int, int]:
    """Calculates a square crop box based on heatmap relevance."""
    row1, row2, col1, col2 = get_crop_range(heatmap, crop_th)

    dr = row2 - row1
    dc = col2 - col1
    if dr > dc:
        col1 -= (dr - dc) // 2
        col2 += (dr - dc) // 2
        if col1 < 0:
            col2 -= col1
            col1 = 0
    elif dc > dr:
        row1 -= (dc - dr) // 2
        row2 += (dc - dr) // 2
        if row1 < 0:
            row2 -= row1
            row1 = 0

    return row1, row2, col1, col2


@torch.no_grad()
def vis_lighten_img_border(
    data_batch, heatmaps, rf=False, alpha=0.4, vis_th=0.02, crop_th=0.01, kernel_size=51
) -> Image.Image:
    """
    Visualize images with lightened borders based on relevance heatmaps.

    This function creates visualizations by lightening regions with low relevance
    scores, making high-relevance areas more prominent. It can optionally crop
    images to focus on relevant regions and applies Gaussian blur for smoothing.

    Parameters
    ----------
    data_batch : torch.Tensor
        Batch of input images of shape (batch_size, channels, height, width).
    heatmaps : torch.Tensor
        Relevance heatmaps of shape (batch_size, height, width).
    rf : bool, default=False
        Whether to crop images to the receptive field of relevant regions.
    alpha : float, default=0.4
        Blending factor for lightening low-relevance regions. Must be in [0, 1].
    vis_th : float, default=0.02
        Visibility threshold for determining relevant regions. Must be in [0, 1).
    crop_th : float, default=0.01
        Cropping threshold for receptive field cropping. Must be in [0, 1).
    kernel_size : int, default=51
        Kernel size for Gaussian blur smoothing of heatmaps.

    Returns
    -------
    list of PIL.Image
        List of processed PIL Images with lightened borders and optional cropping.

    Raises
    ------
    ValueError
        If alpha is not in [0, 1], vis_th not in [0, 1), or crop_th not in [0, 1).
    AssertionError
        If no masking or cropping is applied to any image in the batch,
        which may indicate issues with thresholds or heatmaps.

    Examples
    --------
    >>> import torch
    >>> data = torch.randn(2, 3, 224, 224)
    >>> heatmaps = torch.randn(2, 224, 224)
    >>> images = vis_lighten_img_border(data, heatmaps, alpha=0.3)
    >>> len(images)
    2
    >>> type(images[0])
    <class 'PIL.Image.Image'>
    """
    if alpha > 1 or alpha < 0:
        raise ValueError("'alpha' must be between [0, 1]")
    if vis_th >= 1 or vis_th < 0:
        raise ValueError("'vis_th' must be between [0, 1)")
    if crop_th >= 1 or crop_th < 0:
        raise ValueError("'crop_th' must be between [0, 1)")

    imgs = []
    any_masked = False

    for i in range(len(data_batch)):
        img = data_batch[i]

        filtered_heat = gaussian_blur(heatmaps[i].unsqueeze(0), kernel_size=kernel_size)[0]
        filtered_heat = filtered_heat.abs() / (filtered_heat.abs().max() + 1e-8)
        vis_mask = filtered_heat > vis_th

        if rf:
            row1, row2, col1, col2 = _get_square_crop_box(filtered_heat, crop_th)

            img_t = img[..., row1:row2, col1:col2]
            vis_mask_t = vis_mask[row1:row2, col1:col2]

            if img_t.sum() != 0 and vis_mask_t.sum() != 0:
                img = img_t
                vis_mask = vis_mask_t
                any_masked = True

        inv_mask = ~vis_mask

        # Check if any masking is applied
        if vis_mask.any():
            any_masked = True

        # Lighten the pixels outside the mask
        white_background = torch.ones_like(img)
        img = img * vis_mask + (img * (1 - alpha) + white_background * alpha) * inv_mask

        img = imgify(img.detach().cpu()).convert("RGBA")

        img_ = np.array(img).copy()
        img_[..., 3] = (vis_mask * 255).detach().cpu().numpy().astype(np.uint8)
        img_ = mystroke(Image.fromarray(img_), 1, color="black")

        img.paste(img_, (0, 0), img_)

        imgs.append(img.convert("RGB"))

    if not any_masked:
        raise AssertionError(
            "No masking or cropping was applied to any image in the batch. "
            "This may indicate that the visibility threshold (vis_th) is too high "
            "or that there's an issue with the heatmaps."
        )

    return imgs


@torch.no_grad()
def vis_opaque_img_border(
    data_batch, heatmaps, rf=True, alpha=0.4, vis_th=0.02, crop_th=0.01, kernel_size=51
) -> Image.Image:
    """
    Visualize Dark Image Border.

    Draws reference images. The function lowers the opacity in regions with relevance lower than max(relevance)*vis_th.
    In addition, the reference image can be cropped where relevance is less than max(relevance)*crop_th by setting 'rf'
    to True.

    Parameters
    ----------
    data_batch: torch.Tensor
        original images from dataset without FeatureVisualization.preprocess() applied to it
    heatmaps: torch.Tensor
        ouput heatmap tensor of the CondAttribution call
    rf: boolean
        Computes the CRP heatmap for a single neuron and hence restricts the heatmap to the receptive field.
        The amount of cropping is further specified by the 'crop_th' argument.
    alpha: between [0 and 1]
        Regulates the transparency in low relevance regions.
    vis_th: between [0 and 1)
        Visualization Threshold: Increases transparency in regions where relevance is smaller than max(relevance)*vis_th
    crop_th: between [0 and 1)
        Cropping Threshold: Crops the image in regions where relevance is smaller than max(relevance)*crop_th.
        Cropping is only applied, if receptive field 'rf' is set to True.
    kernel_size: scalar
        Parameter of the torchvision.transforms.functional.gaussian_blur function used to smooth the CRP heatmap.

    Returns
    -------
    image: list of PIL.Image objects
        If 'rf' is True, reference images have different shapes.

    """
    if alpha > 1 or alpha < 0:
        raise ValueError("'alpha' must be between [0, 1]")
    if vis_th >= 1 or vis_th < 0:
        raise ValueError("'vis_th' must be between [0, 1)")
    if crop_th >= 1 or crop_th < 0:
        raise ValueError("'crop_th' must be between [0, 1)")

    imgs = []
    for i in range(len(data_batch)):
        img = data_batch[i]

        filtered_heat = gaussian_blur(heatmaps[i].unsqueeze(0), kernel_size=kernel_size)[0]
        filtered_heat = filtered_heat.abs() / (filtered_heat.abs().max() + 1e-8)
        vis_mask = filtered_heat > vis_th

        if rf:
            row1, row2, col1, col2 = _get_square_crop_box(filtered_heat, crop_th)

            img_t = img[..., row1:row2, col1:col2]
            vis_mask_t = vis_mask[row1:row2, col1:col2]

            if img_t.sum() != 0 and vis_mask_t.sum() != 0:
                # check whether img_t or vis_mask_t is not empty
                img = img_t
                vis_mask = vis_mask_t

        inv_mask = ~vis_mask
        outside = (img * vis_mask).sum((1, 2)).mean(0) / stabilize(vis_mask.sum()) > 0.5

        img = img * vis_mask + img * inv_mask * alpha + outside * 0 * inv_mask * (1 - alpha)

        img = imgify(img.detach().cpu()).convert("RGBA")

        img_ = np.array(img).copy()
        img_[..., 3] = (vis_mask * 255).detach().cpu().numpy().astype(np.uint8)
        img_ = mystroke(Image.fromarray(img_), 1, color="black" if outside else "black")

        img.paste(img_, (0, 0), img_)

        imgs.append(img.convert("RGB"))

    return imgs


def mystroke(img, size: int, color: str = "black"):
    """
    Apply a stroke effect to an image by detecting edges and drawing ellipses around them.
    This function creates a stroke effect by first finding edges in the input image,
    then drawing filled ellipses at edge locations to create an outline effect.
    The original image is then pasted on top of the stroke layer.

    Parameters
    ----------
    img : PIL.Image.Image
        The input image to apply the stroke effect to. Must be a PIL Image object.
    size : int
        The radius of the ellipses used to create the stroke effect. Larger values
        create thicker strokes.
    color : str, optional
        The color of the stroke effect. Accepts "black" for dark strokes or any
        other value for white strokes. Default is "black".

    Returns
    -------
    PIL.Image.Image
        A new image with the stroke effect applied. The returned image maintains
        the same mode and dimensions as the input image.

    Notes
    -----
    The function uses PIL's FIND_EDGES filter to detect edges and creates
    semi-transparent ellipses (opacity 180/255) for the stroke effect.
    Black strokes use RGBA(0, 0, 0, 180) and white strokes use RGBA(255, 255, 255, 180).
    """
    X, Y = img.size
    edge = img.filter(ImageFilter.FIND_EDGES).load()
    stroke = Image.new(img.mode, img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(stroke)
    fill = (0, 0, 0, 180) if color == "black" else (255, 255, 255, 180)
    for x in range(X):
        for y in range(Y):
            if edge[x, y][3] > 0:
                draw.ellipse((x - size, y - size, x + size, y + size), fill=fill)
    stroke.paste(img, (0, 0), img)

    return stroke


@torch.no_grad()
def crop_and_mask_images(data_batch, heatmaps, rf=False, alpha=0.4, vis_th=0.02, crop_th=0.01, kernel_size=51):
    """
    Crop and adjust images based on heatmaps.

    This function processes a batch of images by applying Gaussian blur to their
    corresponding heatmaps, cropping the images based on the filtered heatmaps,
    and converting them to RGB format.

    Parameters
    ----------
    data_batch : list or array-like
        Batch of input images to be processed.
    heatmaps : list or array-like
        Corresponding attention heatmaps for each image in the batch.
    rf : bool, optional
        Receptive field flag (currently unused), by default False.
    alpha : float, optional
        Alpha blending parameter, must be between [0, 1], by default 0.4.
    vis_th : float, optional
        Visibility threshold, must be between [0, 1), by default 0.02.
    crop_th : float, optional
        Cropping threshold for determining crop boundaries, must be between [0, 1),
        by default 0.01.
    kernel_size : int, optional
        Size of the Gaussian blur kernel, by default 51.

    Returns
    -------
    list
        List of processed PIL Images in RGB format, cropped according to their
        respective heatmaps.

    Raises
    ------
    ValueError
        If alpha is not between [0, 1].
    ValueError
        If vis_th is not between [0, 1).
    ValueError
        If crop_th is not between [0, 1).

    Notes
    -----
    The function applies Gaussian blur to normalize heatmaps, determines crop
    boundaries based on the crop threshold, and converts the final images to
    RGB format for visualization.
    """
    if alpha > 1 or alpha < 0:
        raise ValueError("'alpha' must be between [0, 1]")
    if vis_th >= 1 or vis_th < 0:
        raise ValueError("'vis_th' must be between [0, 1)")
    if crop_th >= 1 or crop_th < 0:
        raise ValueError("'crop_th' must be between [0, 1)")

    imgs = []

    for i in range(len(data_batch)):
        img = data_batch[i]

        filtered_heat = gaussian_blur(heatmaps[i].unsqueeze(0), kernel_size=kernel_size)[0]
        filtered_heat = filtered_heat.abs() / (filtered_heat.abs().max())

        # Apply cropping based on the heatmap
        row1, row2, col1, col2 = _get_square_crop_box(filtered_heat, crop_th)

        img = img[..., row1:row2, col1:col2]

        img = imgify(img.detach().cpu()).convert("RGBA")

        imgs.append(img.convert("RGB"))

    return imgs
