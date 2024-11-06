import cv2
import numpy as np
import glob

CHECKERBOARD = (6, 8) 
square_size = 0.025  

obj_points = []  
img_points = []  

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)
objp *= square_size

images = glob.glob('distorted/*.png')
print("found images", images)

for image_file in images:
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    print(f"Image: {image_file}, Corners found: {ret}")

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        img_points.append(corners2)
        obj_points.append(objp)

        cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
        cv2.imshow('Checkerboard', image)
        cv2.waitKey(500)
    else:
        print(f"Failed to find corners in image {image_file}")

cv2.destroyAllWindows()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)
