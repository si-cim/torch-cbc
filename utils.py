import cv2 as cv
import os


def visualize_components(epoch, model, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx, c in enumerate(model.components):
        component = c
        img = component.view(28, 28).cpu().data.numpy()
        img = img * 255
        img = cv.resize(img, (56, 56))

        cv.imwrite(f"{save_path}/{epoch}_{idx}.png",
                   cv.cvtColor(img, cv.COLOR_RGB2BGR))
