import cv2
import numpy as np
from typing import Tuple


def resize_with_pad(image: np.array, 
                    new_shape: Tuple[int, int], 
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image


if __name__ == "__main__":
    video_path = 'walk1.mp4'


    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', None, 30.0, (540, 540))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret : break

        mid = frame.shape[1] / 2 
        size = frame.shape[0]
        frame = frame[:, int(mid-size/2):int(mid+size/2), :]       
        cv2.imshow("image", frame)
        cv2.waitKey(1)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
