import cv2
import numpy as np

def detect_hair_length(face_img):
    h, w, _ = face_img.shape
    
    # Take lower 40% of face (hair near shoulders area)
    lower_part = face_img[int(h*0.6):h, :]
    
    gray = cv2.cvtColor(lower_part, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    if edge_density > 20:
        return "Long"
    else:
        return "Short"
