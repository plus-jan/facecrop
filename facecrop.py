import cv2
import dlib
import sys
from PIL import Image

def detect_and_crop_dog_face(image_path, output_path):
    detector = dlib.cnn_face_detection_model_v1('mmod_dog_hipsterizer.dat')

    image = cv2.imread(image_path)
    if image is None:
        print("Could not read input image")
        return

    dets = detector(image, 1)
    if not dets:
        print("No dog face detected")
        return

    # Choose the first detected dog face
    det = dets[0]

    # Calculate the crop area
    x = max(0, det.rect.left() - 256)
    y = max(0, det.rect.top() - 256)
    width = min(512, image.shape[1] - x)
    height = min(512, image.shape[0] - y)
    crop_area = (x, y, x + width, y + height)

    # Crop and save the image
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cropped_img = img.crop(crop_area)
    cropped_img.save(output_path)
    print(f"Dog face cropped and saved to {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python detect_and_crop_dog_face.py <input_image_path> <output_image_path>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]

    detect_and_crop_dog_face(input_image_path, output_image_path)

