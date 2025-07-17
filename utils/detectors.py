import cv2
from facenet_pytorch import MTCNN
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

def detect_faces(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = mtcnn.detect(rgb)
    boxes = None
    if result is not None:
        if len(result) == 2:
            boxes, _ = result
        elif len(result) == 3:
            boxes, _, _ = result
    regions = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            regions.append((x1, y1, x2, y2))
    return regions
