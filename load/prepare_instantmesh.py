import os
from PIL import Image
import numpy as np
import argparse

def add_random_background(image, bg):
    bg_color = bg  # float
    return add_background(image, bg_color)


def add_background(image_pil, bg_color):
    image = np.array(image_pil)
    rgb, alpha = image[..., :3], image[..., 3:]
    alpha = alpha.astype(np.float32) / 255.0
    image_new = rgb * alpha + bg_color * (1 - alpha)
    return Image.fromarray(image_new.astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--mvimage_path",
    type=str,
    help="select instant3d or instantmesh",
    )
    args = parser.parse_args()

    target_path = "./load/mv_instantmesh"

    image_sets = os.listdir(args.mvimage_path)
    image_sets = [os.path.join(args.mvimage_path, p) for p in image_sets]

    for image_path in image_sets:
        image_name = os.path.basename(image_path)[:-4]
        save_dir = os.path.join(target_path, image_name)
        os.makedirs(save_dir, exist_ok=True)

        supportimg_list = []

        imgp = np.array(Image.open(image_path)).reshape((3, 320, 2, 320, 3))
        imgp = imgp.transpose((0, 2, 1, 3, 4)).reshape((6, 320, 320, 3))
        imgp = [Image.fromarray(p) for p in imgp]
        supportimg_list += imgp

        for i, img in enumerate(supportimg_list):
            img.save(os.path.join(save_dir, '%d.png'%i))