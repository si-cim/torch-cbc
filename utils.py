import os

from PIL import Image, ImageOps
from torchvision.transforms import ToPILImage


def get_concat_h(images):
    dst = Image.new("RGB", (len(images) * (64 + 8), 64 + 8))
    for i, im in enumerate(images):
        dst.paste(im, (i * (64 + 8), 0))
    return dst


def visualize_components(epoch, model, save_path, verbose=True):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    images = []
    for idx, component in enumerate(model.components):
        image = ToPILImage()(component.cpu()).convert("RGB")
        image = image.resize((64, 64))
        # image.save(f"{save_path}/{epoch}_{idx}.png")
        image = ImageOps.expand(image, border=4, fill="black")
        images.append(image)
    result = get_concat_h(images)
    filename = f"{save_path}/{epoch}.png"
    result.save(filename)
    if verbose:
        print(f"Saved components image from epoch {epoch} under {filename}.")
