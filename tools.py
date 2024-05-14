import numpy as np
import cv2

# ближайший сосед
K1 = lambda t: \
    1 if -0.5 <= t < 0.5 \
    else 0


# интерполяция первого порядка
K2 = lambda t: \
    1 - abs(t) if -1 < t < 1 \
    else 0


# сплайновая интерполяция третьего порядка
K3 = lambda t, a: \
    (a + 2) * abs(t)**3 - (a + 3) * abs(t)**2 + 1 if abs(t) <= 1 \
    else a * abs(t)**3 - 5 * a * abs(t)**2 + 8 * a * abs(t) - 4 * t if 1 < abs(t) < 2 \
    else 0

def resize_images(img1, img2):
    # Определяем размер меньшего изображения
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])

    # Приводим оба изображения к одному размеру
    img1_resized = cv2.resize(img1, (min_width, min_height))
    img2_resized = cv2.resize(img2, (min_width, min_height))

    return img1_resized, img2_resized