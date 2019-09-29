import numpy as np
import torch

def iou(box, boxes, mode='inter'):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])

    w = torch.clamp(x2 - x1, min=0)
    h = torch.clamp(y2 - y1, min=0)

    inter = w * h

    if mode == 'inter':
        return inter / (box_area + boxes_area - inter)
    elif mode == 'min':
        return inter / torch.min(box_area, boxes_area)

#
def nms(boxes, thresh=0.3,  mode='inter'):

    keep_boxes = []
    if boxes.size(0) == 0:
        print("125")
        return keep_boxes
    args = boxes[:, 4].argsort(descending=True)
    sort_boxes = boxes[args]

    while len(sort_boxes) > 0:
        _box = sort_boxes[0]
        keep_boxes.append(_box)

        if len(sort_boxes) > 1:
            _boxes = sort_boxes[1:]
            _iou = iou(_box, _boxes, mode)
            sort_boxes = _boxes[_iou < thresh]
        else:
            break

    return torch.stack(keep_boxes)

def convert_to_square(box):
    squareBox = box.clone()

    if box.size(0)== 0:
        return np.array([])

    w = box[:,2] - box[:,0]
    h = box[:,3] - box[:,1]

    maxSide = torch.max(w,h)

    squareBox[:,0] = box[:,0] + w*0.5 - (maxSide*0.5)
    squareBox[:,1] = box[:,1] + h*0.5 - (maxSide*0.5)
    squareBox[:,2] = squareBox[:,0] + maxSide
    squareBox[:,3] = squareBox[:,1] + maxSide

    return squareBox




if __name__ == '__main__':
    a=np.array([10,10,50,70,10])
    b=np.array([[10,10,50,70,10],[10,10,50,70,10],[10,10,50,70,10],[10,10,50,70,10],
                [9,10,50,70,10.1],[9,10,50,70,10],[9,10,50,70,10],[9,10,50,70,10],
                [10,10,50,70,10],[10,10,50,70,10],[10,10,50,70,10],[10,10,50,70,10]])

    print(nms(b))
