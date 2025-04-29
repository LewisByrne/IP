import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

LimgPoints=[]
RimgPoints=[]
objPoints = []
errors = []

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

def findError(imagePoints,rvecs,tvecs,camera_matrix,dist_coeffs):
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

def plotPoints(imagePoints,projected_points):
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
    

def findImgPoints(image,number,side):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 2)
    if number == 1:
        gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 27, 3)  
    else:
        gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 5)
    kernel = np.ones((5, 5), np.uint8)  # Larger kernel to remove noise inside the circle
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
    
    params = cv.SimpleBlobDetector_Params()
    
    if number == 1:
        params.filterByArea = True
        params.minArea = 2600  
        params.maxArea = 4000 
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.filterByInertia = False 
    else:
        params.filterByArea = True
        params.minArea = 3200  
        params.maxArea = 5000 
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.filterByInertia = False 
        
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)


    blobs = cv.drawKeypoints(image, keypoints, None, (0, 0, 255),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    for kp in keypoints:
        center_x, center_y = kp.pt
        if side == 'L':
            LimgPoints.append([center_x, center_y])
        else:
            RimgPoints.append([center_x, center_y])      
    return blobs,gray

objectPoints = np.array(findObjPoints(14.7,16.6),dtype=np.float32)
Limage = cv.imread('L23.png', cv.IMREAD_COLOR)
Rimage = cv.imread('R23.png', cv.IMREAD_COLOR)

with Image.open('L23.png') as img:
    width, height = img.size

Ldetected,Lgray = findImgPoints(Limage,1,'L')    ###L23 = 1, the rest ==2
Rdetected,Rgray = findImgPoints(Rimage,2,'R')


LimagePoints = [np.array(LimgPoints, dtype=np.float32)]
RimagePoints = [np.array(RimgPoints, dtype=np.float32)]
objPoints.append(objectPoints)

flags = 1 + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_FIX_ASPECT_RATIO 

F=12000


K = np.array([
    [F,  0,  width/2], 
    [ 0, F,  height/2], 
    [ 0,  0,   1]
], dtype=np.float32)


Lret, Lcamera_matrix, Ldist_coeffs, Lrvecs, Ltvecs = cv.calibrateCamera(objPoints, LimagePoints, Lgray.shape[::-1], K,None, flags=flags)
Lprojected_points, Lerror = findError(LimagePoints,Lrvecs,Ltvecs,Lcamera_matrix,Ldist_coeffs)
plotPoints(LimagePoints,Lprojected_points)

print(Lcamera_matrix)
print("")


Rret, Rcamera_matrix, Rdist_coeffs, Rrvecs, Rtvecs = cv.calibrateCamera(objPoints, RimagePoints, Rgray.shape[::-1], K,None, flags=flags)
Rprojected_points, Rerror = findError(RimagePoints,Rrvecs,Rtvecs,Rcamera_matrix,Rdist_coeffs)
plotPoints(RimagePoints,Rprojected_points)


print(Rcamera_matrix)
print("")
print("Left Error is",Lret)
print("Right Error is",Rret)

flags  = None
criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv.stereoCalibrate(
    objPoints, LimagePoints, RimagePoints,
    Lcamera_matrix, Ldist_coeffs, Rcamera_matrix, Rdist_coeffs,
    Lgray.shape[::-1], None, None, None, None,
    flags, criteria)

print("")
print("Stereo Calibration Error is", ret)

print("")
print("Rotaion matrix (Rotation of the right camera with respect to the Left)",)
print(R)

print("")
print("Translation vector (Distance from left camera to right camera)")
print(T)


