import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
import os
from itertools import combinations

imgPoints =[]
objPoints = []
radiusArray = []
errors = []



def removeFiles(name):
    if os.path.exists(name) is False:
        return 
    else:
        os.remove(Str)
        
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

def drawCircles(gray,circle):
    count = 0
    averageRadius = 0
    if circle is not None:
        circle = np.uint16(np.around(circle))
        for i in circle[0, :]:
            center = (i[0], i[1])
            cv.circle(src, center, 1, (0, 0, 255), 2)
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 2)
            averageRadius += radius
            count += 1
    averageRadius = averageRadius/count
    radiusArray.append(averageRadius)
    #cv.imshow("circles", src)
    #cv.waitKey(0)
    return 

def findError():
    meanError = 0
    for i in range(len(objPoints)):
        imgP = np.asarray(imgPoints[i], dtype=np.float32)
        imgpoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        imgpoints2 = np.asarray(imgpoints2, dtype=np.float32)
        imgpoints2 = imgpoints2.reshape(-1, 2) 
        error = cv.norm(imgP, imgpoints2, cv.NORM_L2)/len(imgpoints2)
        meanError += error
        errors.append(meanError)
        error = (meanError/len(objPoints))
    #print( "Total error is",np.round(error,4) )
    return imgpoints2,error

def plotPoints(imgPoints,projected_points):
    imgPoints = np.array(imgPoints)
    imgPoints = np.squeeze(imgPoints, axis=0)
    projected_points = np.array(projected_points)
    plt.figure(figsize=(8,5))
    plt.scatter(imgPoints[:, 0], imgPoints[:, 1], color='blue', label='Detected Image Points', s=30)
    plt.scatter(projected_points[:, 0], projected_points[:, 1], color='red', label='Projected Points', s=30, alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(-0.08, 1.15), borderaxespad=0.)
    plt.axis('equal')
    plt.grid()
    plt.title("Detected vs Projected Points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
def showVaribles():
    print("")
    print("The Best Results Are")
    print("Camera matrix:\n", bestMatrix)
    print("")
    print("Distortion coefficients:\n", bestCoeff)
    print("")
    
    apertureWidth = 7.4*10**-3 * 1600
    apertureHeight = 7.4*10**-3 * 1200
    fov_x, fov_y, focal_length, principal_point, aspect_ratio =cv.calibrationMatrixValues(bestMatrix, (width,height), apertureWidth, apertureHeight)

    print("Focal Length is",np.round(focal_length,2),"mm")
    print("The Principle point is",np.round(principal_point,2),"mm")
    print("")
    
    print("RMS Error is", best_error)
    print("Mean Error is", meanError)



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
        
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 1e-12)
    return flag_combinations,criteria
   
    
def numberPoints(imgArray, best_flag):
    errors = []
    points = []
    objArray = np.array(findObjPoints(14.661,16.637))
    imgArray = np.array(imgArray).reshape(-1, 2)
    n = 13

    
    flag_value = sum(best_flag)
    for j in range(len(objArray)-(n-1)):
        objPoints = objArray[:j+n]
        imgPoints = imgArray[:j+n]
        objPoints = np.array(objPoints,dtype=np.float32)
        objPoints = [objPoints]
        imgPoints = np.array(imgPoints,dtype=np.float32)
        imgPoints = [imgPoints]

        K = cv.initCameraMatrix2D(objPoints, imgPoints, gray.shape[::-1])
        K[0, 0] = K1  
        K[1, 1] = K2  
        K[0, 2] = K3  
        K[1, 2] = K4
            
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], K, None,flags = flag_value,criteria=criteria)
        points.append(j+n)
        
        meanError = 0
        for i in range(len(objPoints)):
            imgP = np.asarray(imgPoints[i], dtype=np.float32)
            imgpoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            imgpoints2 = np.asarray(imgpoints2, dtype=np.float32)
            imgpoints2 = imgpoints2.reshape(-1, 2) 
            error = cv.norm(imgP, imgpoints2, cv.NORM_L2)/len(imgpoints2)
            meanError += error

        error = (meanError/len(objPoints))
        errors.append(error)
   
    
    plt.figure(3)
    plt.plot(points, errors,'rx--')
    plt.xlabel('Number of Points')
    plt.ylabel('Error')
    plt.title('Error vs Number of Points')
    plt.grid()
    plt.show()    
    
    toRefine = []
    dy = np.diff(errors)
    threshold = 8
    large_jumps = dy > threshold
    for i in range(len(large_jumps)):
        if large_jumps[i] == True:
            toRefine.append(n+i)
    print("Points which cause Significant Error", toRefine)
    return errors,points,toRefine    
   

def refinePoints(imgArray,toRefine,best_flags): 
    flag_value = sum(best_flags)
    n=13
    objArray = np.array(findObjPoints(14.661,16.637))
    imgArray = np.array(imgArray).reshape(-1, 2)
    refinedObjArray = np.delete(objArray,toRefine, axis=0)
    refinedImgArray = np.delete(imgArray,toRefine, axis=0)
    
    refinedObjPoints = [np.array(refinedObjArray,dtype=np.float32)]
    refinedImgPoints = [np.array(refinedImgArray,dtype=np.float32)]
    
    K = cv.initCameraMatrix2D(refinedObjPoints, refinedImgPoints, gray.shape[::-1])
    K[0, 0] = K1  
    K[1, 1] = K2  
    K[0, 2] = K3  
    K[1, 2] = K4
 
  
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(refinedObjPoints, refinedImgPoints, gray.shape[::-1], K, None,flags = flag_value,criteria=criteria)
    
    meanError = 0
    for i in range(len(refinedObjPoints)):
        imgP = np.asarray(refinedImgPoints[i], dtype=np.float32)
        imgpoints2, _ = cv.projectPoints(refinedObjPoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        imgpoints2 = np.asarray(imgpoints2, dtype=np.float32)
        imgpoints2 = imgpoints2.reshape(-1, 2) 
        error = cv.norm(imgP, imgpoints2, cv.NORM_L2)/len(imgpoints2)
        meanError += error
    error = (meanError/len(refinedObjPoints))
    print("Refined Error is", error)
    
    
    
    errors = []
    points = []
    for j in range(len(refinedObjArray)-(n-1)):
        objPoints = refinedObjArray[:j+n]
        imgPoints = refinedImgArray[:j+n]
        objPoints = [np.array(objPoints,dtype=np.float32)]
        imgPoints = [np.array(imgPoints,dtype=np.float32)]
        
        K = cv.initCameraMatrix2D(objPoints, imgPoints, gray.shape[::-1])
        K[0, 0] = K1  
        K[1, 1] = K2  
        K[0, 2] = K3  
        K[1, 2] = K4
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], K, None,flags = flag_value,criteria=criteria)
        points.append(j+n)
        
        meanError = 0
        for i in range(len(objPoints)):
            imgP = np.asarray(imgPoints[i], dtype=np.float32)
            imgpoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            imgpoints2 = np.asarray(imgpoints2, dtype=np.float32)
            imgpoints2 = imgpoints2.reshape(-1, 2) 
            error = cv.norm(imgP, imgpoints2, cv.NORM_L2)/len(imgpoints2)
            meanError += error
        error = (meanError/len(objPoints))
        errors.append(error)
    
    
    plt.figure(3)
    plt.plot(points, errors,'rx--')
    plt.xlabel('Number of Points')
    plt.ylabel('Error')
    plt.title('Refined Error vs Number of Points')
    plt.grid()
    plt.show() 
    
    apertureWidth = 7.4*10**-3 * 1600
    apertureHeight = 7.4*10**-3 * 1200
    fov_x, fov_y, focal_length, principal_point, aspect_ratio =cv.calibrationMatrixValues(camera_matrix, (width,height), apertureWidth, apertureHeight)

    print("Refined Focal Length is",np.round(focal_length,2),"mm")
    
    return
          

for i in range(20):
    Str = str(i+1) + ".png"
    removeFiles(Str)  
 

images = glob.glob('L23.png')
objectPoints = np.array(findObjPoints(14.661,16.637),dtype=np.float32)


for filename in images:
        src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT,0.2, rows / 12, param1=30, param2=33, minRadius=32, maxRadius=37)
        drawCircles(gray,circles)
        
        if circles is not None:
            circles = np.uint16(np.around(circles[0, :]))
            circles = sorted(circles, key=lambda c: (c[1], c[0]))
            imagePoints = np.array([[c[0], c[1]] for c in circles], dtype=np.float32)
            imgPoints.append(imagePoints)
            objPoints.append(objectPoints)
            
        with Image.open(filename) as img:
            width, height = img.size
             


aspect_ratio = 1.0
flag_combinations, criteria = findFlags()  

best_error =float('inf')
rmsGuessError = []
guesses = []

g=6000
for i in range(50):
    guesses.append(g)
    g =g+200
    
for guess in guesses:
        flag_value = sum([1,8, 16384, 32768, 262144])
        #flag_value = sum([1,4, 8, 16384, 32768, 262144])
        #flag_value = sum([1,16384, 32768, 262144])
        #flag_value = sum([1,4, 2, 8, 16384, 32768])
        
        
        K = cv.initCameraMatrix2D(objPoints, imgPoints, gray.shape[::-1])
        K[0, 0] = guess  
        K[1, 1] = guess 
        K[0, 2] = width/2 
        K[1, 2] = height/2
        K1 = guess  
        K2= guess 
        K3 = width/2 
        K4 = height/2

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], K, None,flags=flag_value)
        
        
        
        if ret is not None:
            projected_points, error = findError()
            plotPoints(imgPoints,projected_points)
            rmsGuessError.append(ret)
            
            if ret < best_error:
               best_error = ret
               meanError = error
               bestMatrix = camera_matrix
               bestCoeff = dist_coeffs
               bestPoints = projected_points
               
           
    
showVaribles()

plt.figure(3)
plt.plot(guesses, rmsGuessError,'rx--')
plt.xlabel('Fx/Fy Inital Guess')
plt.ylabel('RMS Error')
plt.title('RMS Error Vs inital Guess')
plt.grid()
plt.show()
plotPoints(imgPoints,bestPoints)



