import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image


imgPoints =[]
objPoints = []
radiusArray = []
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

def drawCircles(gray,circle):
    bA = 0
    sA = np.inf
    Areas = []
    if circle is not None:
        circle = np.uint16(np.around(circle))
        for i in circle[0, :]:
            center = (i[0], i[1])
            cv.circle(src, center, 1, (0, 0, 255), 2)
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 2)
            area = np.pi * radius**2
            Areas.append(area)
            if area > bA:
                bA = area
            if area < sA:
                sA = area
    
    variance = np.var(Areas)
    sd = np.sqrt(variance)

    #print("Average Area is",np.mean(Areas))
    #print("Largest Area is", bA)
    #print("Smallest Area is", sA)   
    #print("Standard Deviation is",sd)     
           
    cv.imwrite("circles.png", src)
    cv.waitKey(0)
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
    #print( "Total Mean error is",np.round(error,4) )
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
    plt.xlabel("x (Pixels)")
    plt.ylabel("y (Pixels)")
    plt.show()
    
def showVaribles():
    print("")
    print("The Best Results Are")
    print("Camera matrix:\n", bestMatrix)
    print("")
    #print("Distortion coefficients:\n", bestCoeff)
    #print("")
    #print("Rotation coeffients:/n", rvecs)
    #print("")
    #print("Translation coeffients:/n", tvecs)
    
    apertureWidth = 7.4*10**-3 * width
    apertureHeight = 7.4*10**-3 * height

    fov_x, fov_y, focal_length, principal_point, aspect_ratio =cv.calibrationMatrixValues(bestMatrix, (width,height), apertureWidth, apertureHeight)

    print("Focal Length is",np.round(focal_length,2),"mm")
    print("The Principle point is",np.round(principal_point,2),"mm")
    print("")
    
    print("Mean Error is",error )
    print("RMS Error is",ret)
    
   


 

images = glob.glob('R23.png')
objectPoints = np.array(findObjPoints(14.7,16.6),dtype=np.float32)


for filename in images:
        src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT,0.18, rows / 12, param1=30, param2=33, minRadius=32, maxRadius=37)
        drawCircles(gray,circles)
        
        if circles is not None:
            circles = np.uint16(np.around(circles[0, :]))
            circles = sorted(circles, key=lambda c: (c[1], c[0]))
            imagePoints = np.array([[c[0], c[1]] for c in circles], dtype=np.float32)
            imgPoints.append(imagePoints)
            objPoints.append(objectPoints)
            
        with Image.open(filename) as img:
            width, height = img.size



ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None,None,None,None)
if ret:
        projected_points, error = findError()
        best_error = ret
        bestMatrix = camera_matrix
        bestCoeff = dist_coeffs
        bestPoints = projected_points

           

showVaribles()
plotPoints(imgPoints,bestPoints)


