import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove

images = os.listdir('photos')
print("Обработка картинок началась.")
print("Убираем фон.")
for image_name in images:
    print(f"Обрабатываем картинку: {image_name}")
    input_path = f'photos/{image_name}'
    output_path = f'rembg/{image_name}'
    input = Image.open(input_path)
    output = remove(input, bgcolor=(255, 255, 255, 255))
    rmbg_image = output.convert("RGB")
    rmbg_image.save(output_path)

images = os.listdir('rembg')
print("Меняем масштаб картинок для онлайн-аптеки.")
for image_name in images:
    print(f"Обрабатываем картинку: {image_name}")
    input_path = f'rembg/{image_name}'
    output_path = f'result/{image_name}'
    input = Image.open(input_path)
    image_cv = cv2.cvtColor(np.array(input), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cropped_resized_image = image_cv
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image_cv[y:y + h, x:x + w]
        scale_x = image_cv.shape[1] / w
        scale_y = image_cv.shape[0] / h
        scale = min(scale_x, scale_y)
        dsize = (int(w * scale), int(h * scale))
        resized_image = cv2.resize(cropped_image, dsize)
        output_image = 255 * np.ones(shape=[image_cv.shape[0], image_cv.shape[1], 3], dtype=np.uint8)
        x_offset = (image_cv.shape[1] - resized_image.shape[1]) // 2
        y_offset = (image_cv.shape[0] - resized_image.shape[0]) // 2

        output_image[y_offset:y_offset + resized_image.shape[0],
        x_offset:x_offset + resized_image.shape[1]] = resized_image
        cropped_resized_image = output_image

    cv2.imwrite(output_path, cropped_resized_image)





