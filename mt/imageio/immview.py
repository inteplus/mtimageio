#!/usr/bin/python3

from mt import tp, logg, cv, np


__all__ = ["immview"]


def get_image(imm):
    """Produces BGR image for display using OpenCV."""
    if imm.pixel_format in ["gray", "bgr"]:
        return imm.image

    if imm.pixel_format == "rgb":
        return np.ascontiguousarray(np.flip(imm.image, axis=-1))

    if imm.pixel_format == "rgba":
        h, w = imm.image.shape[:2]
        image = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        image[h : h * 2, :w, 0] = imm.image[:, :, 2]
        image[h : h * 2, :w, 1] = imm.image[:, :, 1]
        image[h : h * 2, :w, 2] = imm.image[:, :, 0]

        image[:h, w : w * 2, 0] = imm.image[:, :, 3]

        image[:h, :w, 0] = np.round(
            imm.image[:, :, 2].astype(float) * imm.image[:, :, 3].astype(float) / 255
        ).astype(np.uint8)
        image[:h, :w, 1] = np.round(
            imm.image[:, :, 1].astype(float) * imm.image[:, :, 3].astype(float) / 255
        ).astype(np.uint8)
        image[:h, :w, 2] = np.round(
            imm.image[:, :, 0].astype(float) * imm.image[:, :, 3].astype(float) / 255
        ).astype(np.uint8)
        return image

    if imm.pixel_format == "bgra":
        h, w = imm.image.shape[:2]
        image = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        image[h : h * 2, :w, 0] = imm.image[:, :, 0]
        image[h : h * 2, :w, 1] = imm.image[:, :, 1]
        image[h : h * 2, :w, 2] = imm.image[:, :, 2]

        image[:h, w : w * 2, 0] = imm.image[:, :, 3]

        image[:h, :w, 0] = np.round(
            imm.image[:, :, 0].astype(float) * imm.image[:, :, 3].astype(float) / 255
        ).astype(np.uint8)
        image[:h, :w, 1] = np.round(
            imm.image[:, :, 1].astype(float) * imm.image[:, :, 3].astype(float) / 255
        ).astype(np.uint8)
        image[:h, :w, 2] = np.round(
            imm.image[:, :, 2].astype(float) * imm.image[:, :, 3].astype(float) / 255
        ).astype(np.uint8)
        return image

    raise ValueError(
        "Imm with pixel format '{}' is not supported.".format(imm.pixel_format)
    )


def view(image, max_width=640, as_ansi=True):
    """Displays a BGR image."""
    if max_width < image.shape[1]:
        height = image.shape[0] * max_width // image.shape[1]
        image = cv.resize(image, dsize=(max_width, height))
    if as_ansi:
        img2 = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        print(cv.to_ansi(img2))
    else:
        cv.namedWindow("image")
        print("Press any key to exit.")
        cv.imshow("image", image)
        cv.waitKey(0)


def immview(
    imm: cv.Image,
    use_highgui: bool = False,
    max_width: int = 640,
    filepath: tp.Optional[str] = None,
    logger: tp.Optional[logg.IndentedLoggerAdapter] = None,
):
    """Views an image with metadata, either via OpenCV's highgui or on the terminal.

    Parameters
    ----------
    imm : mt.cv.Image
        an image with metadata
    use_highgui : bool
        whether to use OpenCV's highgui or the terminal
    max_width : int
        the maximum width. Only valid if `use_highgui` is True
    filepath : str, optional
        the filepath to the imm
    logger : mt.logg.IndentedLoggerAdapter, optional
        logger for printing purposes

    """
    if logger:
        if filepath:
            logger.info("Image path: {}".format(filepath))
        logger.info("Pixel format: {}".format(imm.pixel_format))
        logger.info("Resolution: {}x{}".format(imm.image.shape[1], imm.image.shape[0]))
        logger.info("Meta:")
        logger.info(imm.meta)
    view(get_image(imm), max_width=max_width, as_ansi=not use_highgui)
