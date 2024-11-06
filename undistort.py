import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

camera_matrix = np.array([[234.145229, 0.330551, 370.627918],
                          [0.000000, 236.049533, 364.741985],
                          [0.000000, 0.000000, 1.000000]])

dist_coeffs = np.array([-0.022016, -0.006007, 0.001086, -0.000218])
current_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(current_dir, 'distorted')
output_folder = os.path.join(current_dir, 'undistorted')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

original_images = []
undistorted_images = []

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    
    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, dist_coeffs, (w, h), np.eye(3), balance=0.0
        )

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2
        )

        undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        undistorted_rgb = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)

        output_path = os.path.join(output_folder, f'undistorted_{image_file}')
        cv2.imwrite(output_path, undistorted_image)

        original_images.append(image_rgb)
        undistorted_images.append(undistorted_rgb)

n_images = len(original_images)
if n_images > 0:
    for i in range(0, n_images, 2):
        current_group = min(2, n_images - i)
        
        fig, axes = plt.subplots(current_group, 2, figsize=(10, 5*current_group))
        
        if current_group == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(current_group):
            image_idx = i + idx
            axes[idx, 0].imshow(original_images[image_idx])
            axes[idx, 0].set_title(f'Original Image {image_idx+1}')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(undistorted_images[image_idx])
            axes[idx, 1].set_title(f'Undistorted Image {image_idx+1}')
            axes[idx, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
else:
    print("Failed to find any images to process.")