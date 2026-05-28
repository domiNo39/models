import cv2
import numpy as np
import ncnn

IMG_PATH = "test.jpeg" 
MODEL_PARAM = "student_int8.param"  
MODEL_BIN = "student_int8.bin"
INPUT_SIZE = 320                    

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    net = ncnn.Net()
    net.load_param(MODEL_PARAM)
    net.load_model(MODEL_BIN)
    img_raw = cv2.imread(IMG_PATH)
    if img_raw is None: return
    h_raw, w_raw = img_raw.shape[:2]
    mat_in = ncnn.Mat.from_pixels_resize(
        img_raw, ncnn.Mat.PixelType.PIXEL_BGR, w_raw, h_raw, INPUT_SIZE, INPUT_SIZE
    )
    mean = [103.53, 116.28, 123.675]
    norm = [0.017429, 0.017507, 0.017124]
    mat_in.substract_mean_normalize(mean, norm)
    ex = net.create_extractor()
    ex.input("in0", mat_in)
    ret, mat_out = ex.extract("out0")
    data = np.array(mat_out)
    if len(data.shape) == 1:
        data = data.reshape((data.shape[0] // 33, 33))
    candidates = []
    strides = [8, 16, 32, 64]
    layers = []
    curr = 0
    for s in strides:
        cnt = (INPUT_SIZE // s) ** 2
        layers.append((s, INPUT_SIZE // s, curr, curr + cnt))
        curr += cnt
    
    scale_x = w_raw / INPUT_SIZE
    scale_y = h_raw / INPUT_SIZE
    for i, row in enumerate(data):
        score = sigmoid(row[4]) * sigmoid(np.max(row[5:]))
        if score > 0.01:
            for (stride, grid, start, end) in layers:
                if start <= i < end:
                    rel_idx = i - start
                    gx = rel_idx % grid
                    gy = rel_idx // grid
                    cx = int((gx * stride + stride/2) * scale_x)
                    cy = int((gy * stride + stride/2) * scale_y)
                    candidates.append((score, cx, cy, stride))
                    break

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_k = candidates[:20] 
    draw_img = img_raw.copy()
    for k in range(3):
        if k < len(top_k):
            print(f"{k+1}: {top_k[k][0]:.4f}")

    for rank, (score, x, y, stride) in enumerate(top_k):
        intensity = int(255 * (rank / 20.0))
        color = (0, 255 - intensity, intensity) 
        radius = int(stride * 0.4)
        cv2.circle(draw_img, (x, y), radius, color, 2)
        cv2.putText(draw_img, str(rank+1), (x-5, y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    filename = f"top20_{INPUT_SIZE}.jpg"
    cv2.imwrite(filename, draw_img)

if __name__ == "__main__":
    main()
