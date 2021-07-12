import cv2 as cv
import torch
import numpy as np
from matplotlib import pyplot as plt


def zys(n):
    value = []
    i = 2
    m = n
    while i <= int(m / 2 + 1) and n != 1:
        print(i, n)
        if n % i == 0:
            value.append(i)
            n = n // i
            i -= 1
        i += 1
    value.append(1)
    if len(value) == 1:
        value.append(m)
    return value


def vis_img(img, *args):
    if img == None:
        pass
    output = img
    l = zys(len(output))
    l1 = 1
    l2 = 1
    tag = 0
    for i in l:
        if tag:
            l1 *= i
            tag = 0
        else:
            l2 *= i
            tag = 1
    print(l1, l2)
    p = 0
    for i, out in enumerate(output):
        b = torch.min(out).item()
        c = 255 / max((torch.max(out).item() - b), b)
        out = (out - b) * c
        out = out.detach().numpy().astype(np.uint8)
        plt.subplot(l2, l1, i + 1)
        plt.imshow(out, "viridis")
        if len(args) == None:
            plt.title('img' + str(i))
        else:
            plt.title(str(args[p]))
            p += 1
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.show()
