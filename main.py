import base64
import os
import sys

import cv2
from text_detection_craft import get_points_from_file, Point, text_detection_module

if __name__ == '__main__':

    img_path = f"C:\\Users\\hansheng\\OneDrive\\Documents\\Lightshot\\Screenshot_98.png"
    use_cuda = False
    try:
        print("Using {}".format(sys.argv[1]))
        use_cuda = True
    except:
        pass

    # CRAFT
    img_list, _ = text_detection_module(img_path, use_cuda)

    image_path = "result"
    output_path = "./output"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for file in os.listdir(image_path):
        if file.endswith(".txt"):
            with open(os.path.join(image_path, file)) as f:
                bboxes = get_points_from_file(f.readlines())
            img_path = os.path.join(img_path)
            im = cv2.imread(img_path)
            for i, box in enumerate(bboxes):
                (h, w) = im.shape[:2]
                (cX, cY) = (w//2, h//2)
                center = Point(cX, cY)
                angle = box.get_rotate_angle()
                M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
                rotated = cv2.warpAffine(im, M, (w, h))
                rotated_tl = box.tl.get_rotated(angle, center)
                rotated_tr = box.tr.get_rotated(angle, center)
                rotated_br = box.br.get_rotated(angle, center)
                rotated_bl = box.bl.get_rotated(angle, center)
                top_left = rotated_tl.y if rotated_tl.y >= 0 else 0
                top_right = rotated_tr.y if rotated_tr.y >= 0 else 0
                top = max(top_left, top_right)
                bot_left = rotated_bl.y if rotated_bl.y <= h else h
                bot_right = rotated_br.y if rotated_br.y <= h else h
                bot = max(bot_right, bot_left)
                cropped = rotated[top:bot, 0:w]

                output = file.replace(".txt", f"_{i}.jpg")
                path = os.path.join(output_path, output)
                try:
                    cv2.imwrite(path, cropped)

                    # Convert to base64 and append
                    _, buffer = cv2.imencode('.jpg', cropped)
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                    print(image_base64)
                except:
                    pass
