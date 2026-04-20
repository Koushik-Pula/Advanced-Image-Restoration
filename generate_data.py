import os
import cv2
import numpy as np
from tqdm import tqdm

def add_motion_blur(image):
    size = np.random.randint(10, 25)
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size) / size
    return cv2.filter2D(image, -1, kernel)

def add_synthetic_haze(image):
    h, w, _ = image.shape
    depth_map = np.linspace(0.8, 0.1, h).reshape(h, 1)
    depth_map = np.repeat(depth_map, w, axis=1)[..., np.newaxis]
    beta = np.random.uniform(1.2, 2.5)
    transmission = np.exp(-beta * depth_map)
    hazy = image * transmission + (0.8 * 255) * (1 - transmission)
    return np.clip(hazy, 0, 255).astype(np.uint8)

def add_synthetic_rain(image):
    h, w, _ = image.shape
    rain_layer = np.zeros((h, w), dtype=np.uint8)
    for _ in range(np.random.randint(500, 2000)):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        length = np.random.randint(10, 30)
        cv2.line(rain_layer, (x, y), (x, y+length), 255, 1)
        
    M = cv2.getRotationMatrix2D((w/2, h/2), angle=np.random.randint(10, 30), scale=1)
    rain_layer = cv2.warpAffine(rain_layer, M, (w, h))
    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    rain_colored = cv2.cvtColor(rain_layer, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 0.7, rain_colored, 0.4, 0)

def build_dataset(clean_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg'))]
    print(f"Building Multi-Degradation Dataset ({len(files)} images)...")
    
    for f in tqdm(files):
        img = cv2.imread(os.path.join(clean_dir, f))
        if img is None: continue
        
        degraded = img.copy()
        if np.random.rand() > 0.5: degraded = add_motion_blur(degraded)
        if np.random.rand() > 0.5: degraded = add_synthetic_haze(degraded)
        if np.random.rand() > 0.5: degraded = add_synthetic_rain(degraded)
        if np.array_equal(degraded, img): degraded = add_synthetic_rain(degraded) # Fallback
            
        cv2.imwrite(os.path.join(output_dir, f), degraded)

if __name__ == "__main__":
    build_dataset("data/DIV2K_train_HR", "data/train_mixed_degraded")