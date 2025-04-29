import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


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
        imgP = np.asarray(SHIFTimagePoints[i], dtype=np.float32)
        imgpoints2, _ = cv.projectPoints(objPoints[i], r_vecs[i], t_vecs[i], camera_matrix, dist_coeffs)
        imgpoints2 = np.asarray(imgpoints2, dtype=np.float32)
        imgpoints2 = imgpoints2.reshape(-1, 2) 
        error = cv.norm(imgP, imgpoints2, cv.NORM_L2)/len(imgpoints2)
        meanError += error
    error = meanError/len(objPoints)
    return imgpoints2,error


def numberPoints(imgArray):
    errors = []
    points = []
    pufx =[]
    pufy =[]
    fxa = []
    objArray = np.array(findObjPoints(14.7,16.6))
    imgArray = np.array(imgArray).reshape(-1, 2)
    K = np.array([[Fx,0,Cx],[0,Fy,Cy],[0,0,1]],dtype=np.float32)
    flag = 1
    n = 9
    for j in range(len(objArray)-(n-1)):
        objPoints = objArray[:j+n]
        imgPoints = imgArray[:j+n]
        objPoints = np.array(objPoints,dtype=np.float32)
        objPoints = [objPoints]
        imgPoints = np.array(imgPoints,dtype=np.float32)
        imgPoints = [imgPoints]
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, size, K, None,flags=flag)
        fxv = camera_matrix[0][0]
        fyv = camera_matrix[1][1]
        pu = abs(fxv-Fx)/Fx *100
        pu2 = abs(fyv-Fy)/Fy *100
        points.append(j+n)
        errors.append(ret)
        pufx.append(pu)
        pufy.append(pu2)
        fxa.append(fxv)
        
    

        
    return errors,points

objectPoints = np.array(findObjPoints(14.7,16.6),dtype=np.float32)

width = 1088
height = 1200

Fx = 1.2e4
Fy = 1.2e4
Cx = 540
Cy = 600
k1 = 1e-5
k2 = 1e-9
k3 = 5e-2
p1 = 0
p2 = 0

Ushift = int(Fx * 0.004)
Lshift = int(-Fx * 0.004)

Tx = []
v = -100
for i in range(100):
    Tx.append(v)
    v=v+2

errors = []

for i in range(len(Tx)):
    K = np.array([[Fx,0,Cx],[0,Fy,Cy],[0,0,1]],dtype=np.float32)
    distC = np.array([k1,k2,p1,p2,k3], dtype=np.float32)
    rvec = np.array([0.08163, 3, -0.379])   
    tvec = np.array([i, -30.7608541, 1997.86496194])
    size = (width,height)

    imagePoints, _ = cv.projectPoints(objectPoints, rvec, tvec, K, distC)
    imgPoints = imagePoints.reshape(-1, 2)
    
    simgPoints = []
    for i in range(len(imgPoints)):
            l=1
            if i == 5:
                Ushift = int(Fx * 0.0005)

            elif i == 10:
                Ushift = int(Fx * 0.001)

            elif i == 15:

                Ushift = int(-Fx * 0.002)
            elif i ==20:

                Ushift = int(-Fx * 0.005)
            else: 
                l=0

                 
            if l == 10:
                    x = imgPoints[i,0] + Ushift
                    y = imgPoints[i,1] + Ushift
            else:
                    x = imgPoints[i,0] 
                    y = imgPoints[i,1] 
                    
            simgPoints.append([x,y])
            
    SHIFTimagePoints = [np.array(simgPoints, dtype=np.float32)]
    simgPoints =np.array(simgPoints).reshape(-1, 2)

    
    objPoints = []
    objPoints.append(objectPoints)
    
    ret, camera_matrix, dist_coeffs, r_vecs, t_vecs = cv.calibrateCamera(objPoints, SHIFTimagePoints, size, None,None, None)
        
    projected_points, error = findError()
   
    plt.figure(figsize=(10, 6))
    plt.scatter(imgPoints[:, 0], imgPoints[:, 1], color='green',  label='True Image Point')
    plt.scatter(projected_points[:, 0], projected_points[:, 1], color='red', label='Projected Points')
    plt.title("Simulated Image Points Vs Projected Points")
    plt.legend(loc='upper left', bbox_to_anchor=(-0.08, 1.15), borderaxespad=0.)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
        
    #print("The Camera Matrix is:")
    #print(camera_matrix)
    #print("")
    #print("Distortion Coeffients are:")
    #print(dist_coeffs)
    #print("")
    #print("The RMS error is",ret)
    #print("The Mean Error is", error)
    
    errors.append(ret)
        
plt.figure(3)
plt.plot(Tx, errors,'rx--',)
plt.xlabel('Tx (pixels)')
plt.ylabel('RMS Error')
plt.title('RMS Error Vs Horizontal Translation')
plt.legend()
plt.grid()
plt.show()