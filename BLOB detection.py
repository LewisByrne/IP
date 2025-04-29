import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

imgPoints=[]
objPoints = []
errors = []

K1 = 15105.18

K2 = 14817.14

K3 = 671.50

K4 = 599.50



def findObjPoints(xDistance,yDistance):
    x0 = 0
    y0 = 0
    rows = 5
    columns = [6,5,6,5,6]
    x = []
    y = []
    xp = [x0,x0 + xDistance/2,x0,x0 + xDistance/2,x0]
    yp = y0
    for i in range(rows):
        xValue = xp[i]
        yValue = yp
        x.append(xValue)
        y.append(yValue)
        for j in range(columns[i]-1):
            xValue = xValue + xDistance
            yValue = yp
            x.append(xValue)
            y.append(yValue)
        yp = yp + yDistance
    
    xy =[]
    for i in range(len(x)):
        xy.append([x[i],y[i],0])
    return xy

def findError():
    meanError = 0
    for i in range(len(objPoints)):
        imgP = np.asarray(imagePoints[i], dtype=np.float32)
        imgpoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        imgpoints2 = np.asarray(imgpoints2, dtype=np.float32)
        imgpoints2 = imgpoints2.reshape(-1, 2) 
        error = cv.norm(imgP, imgpoints2, cv.NORM_L2)/len(imgpoints2)
        meanError += error
        errors.append(meanError)
        error = (meanError/len(objPoints))
    #print( "Total Mean error is",np.round(error,4) )
    return imgpoints2,error

def plotPoints(imgPoints,projected_points):
    imgPoints = np.array(imagePoints)
    imgPoints = np.squeeze(imgPoints, axis=0)
    projected_points = np.array(projected_points)
    plt.figure(figsize=(8,5))
    plt.scatter(imgPoints[:, 0], imgPoints[:, 1], color='blue', label='Detected Image Points', s=30)
    plt.scatter(projected_points[:, 0], projected_points[:, 1], color='red', label='Projected Points', s=30, alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(-0.08, 1.15), borderaxespad=0.)
    plt.axis('equal')
    plt.grid()
    plt.title("Detected vs Projected Points")
    plt.xlabel("x (Pixels)")
    plt.ylabel("y (Pixels)")
    plt.show()

def findFlags():
    all_flags = [
        cv.CALIB_FIX_PRINCIPAL_POINT,
        cv.CALIB_FIX_ASPECT_RATIO,
        cv.CALIB_ZERO_TANGENT_DIST,
        cv.CALIB_RATIONAL_MODEL,
        cv.CALIB_THIN_PRISM_MODEL,
        cv.CALIB_TILTED_MODEL,
    ]
  
    
    flag_combination = []
    for r in range(1, len(all_flags) + 1):
        flag_combination += list(combinations(all_flags, r))
    
    flag_combinations =[]
    for i in flag_combination:
        flag_combinations.append(i + (cv.CALIB_USE_INTRINSIC_GUESS,))
        
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 500, 1e-12)
    return flag_combinations,criteria

objectPoints = np.array(findObjPoints(14.7,16.6),dtype=np.float32)
image = cv.imread('R24.png', cv.IMREAD_COLOR)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (7, 7), 2)
gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 5)  
kernel = np.ones((5, 5), np.uint8)  # Larger kernel to remove noise inside the circle
gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)

params = cv.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 3200  
params.maxArea = 5000 
params.filterByCircularity = True
params.minCircularity = 0.8
params.filterByInertia = False 

detector = cv.SimpleBlobDetector_create(params)
keypoints = detector.detect(gray)


blobs = cv.drawKeypoints(image, keypoints, None, (0, 0, 255),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

bA = 0
sA = np.inf
Areas = []
for kp in keypoints:
    area = kp.size * kp.size * np.pi / 4  
    center_x, center_y = kp.pt
    imgPoints.append([center_x, center_y])
    Areas.append(area)
    if area > bA:
        bA = area
    if area < sA:
        sA= area
    
variance = np.var(Areas)
sd = np.sqrt(variance)

print("Average Area is",np.mean(Areas))
print("Largest Area is", bA)
print("Smallest Area is", sA)   
print("Standard Deviation is",sd)  
print("")       

cv.imwrite("Blobs.png", blobs)
cv.waitKey(0)
cv.destroyAllWindows()

imagePoints = [np.array(imgPoints, dtype=np.float32)]
objPoints.append(objectPoints)


print(gray.shape[::-1])  # (width, height)


K = cv.initCameraMatrix2D(objPoints, imagePoints, gray.shape[::-1])
K[0, 0] = K1  
K[1, 1] = K2  
K[0, 2] = K3  
K[1, 2] = K4

aspect_ratio = 1.0
flag_combination, criteria = findFlags()  

best_error =float('inf')
e = []
i = []
d = 1
for flags in flag_combination:
        flag_value = sum(flags) 

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objPoints, imagePoints, gray.shape[::-1], K, None,flags=flag_value)
        if ret:
            print(d)
            projected_points, error = findError()
            plotPoints(imagePoints,projected_points)
            e.append(ret)
            i.append(d)
            d=d+1
            if ret < best_error:
               best_error = ret
               meanError = error
               bestMatrix = camera_matrix
               bestCoeff = dist_coeffs
               bestPoints = projected_points
               best_flag = flags




print("Camera Matrix is")
print(bestMatrix)
print("")
print("RMS Error is",best_error)
print("Mean Error is", meanError)
print("Best Flags are",best_flag)

plt.figure(2)
plt.plot(i, e,'rx--')
plt.xlabel('Flag Combination')
plt.ylabel('Error')
plt.title('Error vs Flag Combination')
plt.show()
      
plotPoints(imgPoints,bestPoints)
