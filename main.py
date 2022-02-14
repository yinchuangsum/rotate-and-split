import os
import math
import cv2


class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def get_tuple(self):
        return self.x, self.y

    def get_rotated(self, a, o):
        new_x = ((self.x - o.x) * math.cos(-math.radians(a))) - \
                ((self.y - o.y) * math.sin(-math.radians(a))) + o.x
        new_y = ((self.x - o.x) * math.sin(-math.radians(a))) + \
                ((self.y - o.y) * math.cos(-math.radians(a))) + o.y
        return Point(new_x, new_y)

    def __str__(self):
        return f"({self.x}, {self.y})"


class BBox:
    def __init__(self, data):
        coordinates = data.split(',')
        self.tl = Point(coordinates[0], coordinates[1])
        self.tr = Point(coordinates[2], coordinates[3])
        self.bl = Point(coordinates[4], coordinates[5])
        self.br = Point(coordinates[6], coordinates[7])

    @classmethod
    def of(cls, tl, tr, bl, br):
        data = [str(tl.x), str(tl.y), str(tr.x), str(tr.y), str(bl.x), str(bl.y), str(br.x), str(br.y)]
        return BBox(','.join(data))

    def get_center_y(self):
        return (self.tl.y + self.br.y + self.tr.y + self.bl.y) / 4

    def get_rotate_angle(self):
        return math.degrees(math.atan((self.tr.y - self.tl.y) / (self.tr.x - self.tl.x)))

    def get_rotate_box(self, o):
        a = self.get_rotate_angle()
        return BBox.of(self.tl.get_rotated(a, o), self.tr.get_rotated(a, o),
                       self.bl.get_rotated(a, o), self.br.get_rotated(a, o))

    def __str__(self):
        return f"[tl:{self.tl}, tr:{self.tr}, bl:{self.bl}, br:{self.br}]"


def merge_boxes(box_to_merge):
    tl = box_to_merge[0].tl
    tr = box_to_merge[0].tr
    bl = box_to_merge[0].bl
    br = box_to_merge[0].br
    for box in box_to_merge:
        if box.bl.x < bl.x:
            bl = box.bl
        if box.tl.x < tl.x:
            tl = box.tl
        if box.tr.x > tr.x:
            tr = box.tr
        if box.br.x > br.x:
            br = box.br
    return BBox.of(tl, tr, bl, br)


def merge_nearest_bbox(boxes):
    dist = dict()
    points = dict()
    x = []
    for idx, bbox in enumerate(boxes):
        x.append(bbox.get_center_y())
        for i in range(0, idx):
            points[(i, idx)] = abs(x[i] - x[idx])
            distance = abs(x[i] - x[idx])
            if dist.get(distance) is None:
                dist[distance] = []
            dist[distance].append(i)
            dist[distance].append(idx)

    length = len(x)
    box_to_merge = set()
    while length > 3:
        mini = min(dist.keys())
        box_to_merge.update(list(dist.pop(mini)))
        length = len(x) - len(box_to_merge) + 1

    box_to_merge = [boxes[i] for i in box_to_merge]
    [boxes.remove(i) for i in box_to_merge]
    box = merge_boxes(box_to_merge)
    boxes.append(box)

    return boxes


def get_points_from_file(datas):
    boxes = []
    for data in datas:
        if not data.isspace():
            boxes.append(BBox(data.replace("\n", "")))

    if len(boxes) > 3:
        boxes = merge_nearest_bbox(boxes)
    return boxes


if __name__ == '__main__':
    image_path = "images"
    output_path = "./output"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for file in os.listdir(image_path):
        if file.endswith(".txt"):
            with open(os.path.join(image_path, file)) as f:
                bboxes = get_points_from_file(f.readlines())
            img = file.replace(".txt", ".jpg")
            im = cv2.imread(os.path.join(image_path, img))
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
                cv2.imwrite(path, cropped)
