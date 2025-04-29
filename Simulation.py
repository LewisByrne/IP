import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

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

objectPoints = np.array(findObjPoints(14.7,16.6),dtype=np.float32)

width = 1088
height = 1200

Fx = 1.4e4
Fy = 1.4e4
Cx = width/2
Cy = height/2
k1 = 1e-5
k2 = 1e-9
k3 = 5e-2
p1 = 0
p2 = 0

Ushift = int(Fx * 0.004)
Lshift = int(-Fx * 0.004)


K = np.array([[Fx,0,Cx],[0,Fy,Cy],[0,0,1]],dtype=np.float32)
distC = np.array([k1,k2,p1,p2,k3], dtype=np.float32)
rvec = np.array([0.08163, 3, -0.379])   
tvec = np.array([39.14412673, -30.7608541, 1997.86496194])
size = (width,height)

imagePoints, _ = cv.projectPoints(objectPoints, rvec, tvec, K, distC)
imgPoints = imagePoints.reshape(-1, 2)

Kguess = np.array([[1.4e4,0,Cx],[0,1.4e4,Cy],[0,0,1]],dtype=np.float32)
flags= 1

inducedRms = []
inducedMean = []
rmsError = []
meanError = []
fx = []
fy = []
pufx = []
pufy =[]
Ipufx=[]
Ipufy =[]
Irms = []


for j in range(60):
    simgPoints = []
    for i in range(len(imgPoints)):
        if j <= 10:
            l = random.randint(0,2)
            Ushift = int(Fx * 0.0005)
            Lshift = int(-Fx * 0.0005)
        elif j <= 25:
            l = random.randint(0,2)
            Ushift = int(Fx * 0.001)
            Lshift = int(-Fx * 0.001)
        elif j <= 40:
            l = random.randint(0,2)
            Ushift = int(Fx * 0.002)
            Lshift = int(-Fx * 0.002)
        elif j <= 90:
            l = random.randint(0,2)
            Ushift = int(Fx * 0.004)
            Lshift = int(-Fx * 0.004)
        else: 
             l = random.randint(0,2)
             Ushift = int(Fx * 0.006)
             Lshift = int(-Fx * 0.006)
             
        if l == 1:
                x = imgPoints[i,0] + random.randint(Lshift,Ushift)
                y = imgPoints[i,1] + random.randint(Lshift,Ushift)
        else:
                x = imgPoints[i,0] 
                y = imgPoints[i,1] 
        simgPoints.append([x,y])
    SHIFTimagePoints = [np.array(simgPoints, dtype=np.float32)]
    simgPoints =np.array(simgPoints).reshape(-1, 2)
    
    objPoints = []
    objPoints.append(objectPoints)
    
    
    ret, camera_matrix, dist_coeffs, r_vecs, t_vecs = cv.calibrateCamera(objPoints, SHIFTimagePoints, size, Kguess, None, None)
    
    projected_points, error = findError()
    
    
    plt.figure(figsize=(10, 6))
    plt.scatter(imgPoints[:, 0], imgPoints[:, 1], color='green',  label='True Image Point')
    plt.scatter(simgPoints[:, 0], simgPoints[:, 1], color='blue',  label='Image Points')
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
    
    simgPoints = simgPoints.astype(np.float32)
    meanAddedError = 0
    for i in range(len(imgPoints)):
        introducedError = cv.norm(imgPoints[i], simgPoints[i], cv.NORM_L2)
        meanAddedError += introducedError 
    introducedError = meanAddedError/len(imgPoints)
    #print("Introudced Mean Error in image points",introducedError)
    
    rmsAddedError = 0
    for i in range(len(imgPoints)):
        introducedrmsError = cv.norm(imgPoints[i], simgPoints[i], cv.NORM_L2)
        rmsAddedError += introducedrmsError**2 
    rmsIntroducedError = (rmsAddedError/len(imgPoints))**0.5
    #print("Introudced RMS Error in image points",rmsIntroducedError)
    #print("")
    fx2 = camera_matrix[0][0]
    fy2 = camera_matrix[1][1]
    pu = abs((Fx-fx2)/Fx)*100
    pu1 = abs((Fy-fy2)/Fy)*100

    fx.append(fx2)
    pufx.append(pu)
    fy.append(fy2)
    pufy.append(pu1)
    inducedRms.append(rmsIntroducedError)
    inducedMean.append(introducedError)
    rmsError.append(ret)
    meanError.append(error)

    
          
    ret, camera_matrix, dist_coeffs, r_vecs, t_vecs = cv.calibrateCamera(objPoints, SHIFTimagePoints, size, Kguess, None, flags=flags)
          
    fx2 = camera_matrix[0][0]
    fy2 = camera_matrix[1][1]
    pu = abs((Fx-fx2)/Fx)*100
    pu1 = abs((Fy-fy2)/Fy)*100

    
    Ipufx.append(pu)
    Ipufy.append(pu1)
    Irms.append(ret)
    
    
def linear(x,a,b):
    return a*x + b

constants = curve_fit(linear,inducedRms,rmsError)
a_fit = constants[0][0]
b_fit = constants[0][1]
fit = []
for i in inducedRms:
    fit.append(linear(i,a_fit,b_fit))
    
constants = curve_fit(linear,inducedRms,Irms)
a_fit = constants[0][0]
b_fit = constants[0][1]
fit2 = []
for i in inducedRms:
    fit2.append(linear(i,a_fit,b_fit))





plt.figure(1)
plt.plot(inducedRms, rmsError,'rx',label="No intrinsic Guess")
plt.plot(inducedRms, fit,'k-',)
plt.plot(inducedRms,Irms,'bx',label="Intrinsic Guess")
plt.plot(inducedRms, fit2,'k-',)
plt.xlabel('Induced RMS Error of image points')
plt.ylabel('Calibration RMS Error')
plt.title('Calibration Error VS Induced RMS error')
plt.legend()
plt.grid()
plt.show() 



plt.figure(3)
plt.plot(inducedRms, pufx,'rx', label = 'No intrinsic Guess')
plt.plot(inducedRms, Ipufx,'bx', label = 'Intrinsic Guess')
plt.xlabel('Induced RMS Error of image points')
plt.ylabel('%U of Fx')
plt.title('%U of Fx VS Induced RMS error')
plt.legend()
plt.grid()
plt.show() 





plt.figure(3)
plt.plot(inducedRms, pufy,'rx', label = 'No intrinsic Guess')
plt.plot(inducedRms, Ipufy,'bx', label = 'Intrinsic Guess')
plt.xlabel('Induced RMS Error of image points')
plt.ylabel('%U of Fy')
plt.title('%U of Fy VS Induced RMS error')
plt.legend()
plt.grid()
plt.show() 




 




import scipy.stats as stats
      
       
correlation, p_value = stats.pearsonr(pufx, rmsError)
print(f"Fx Correlation Coefficient: {correlation:.3f}, P-value: {p_value:.3f}")

correlation, p_value = stats.pearsonr(pufy, rmsError)
print(f"Fy Correlation Coefficient: {correlation:.3f}, P-value: {p_value:.3f}")