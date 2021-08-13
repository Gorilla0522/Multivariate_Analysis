#Predicting the area burned by wildfires

import csv
import matplotlib.pyplot as plt



#Variable, Constant
X = [] #explanatory valiable  #X = [1,Month,Day,FFMC,DMC,DC,ISI,Temp,RH,Wind,Rain]
B = [] #parameter matrix
Y_area = [] #objective valiable
y = [] #prediced value of Y_area

#Functions
def predict(X,B):
    y = [0 for i in range(len(X))]
    for i in range(len(X)):
        for j in range(len(X[0])):
            y[i] += X[i][j]*B[j]
    return y  
    
def Loss(y,Y):
    E = 0
    for i in range(len(Y)):
        E += (Y[i] - y[i])**2
    return 0.5*E/len(X)

def GradientDescent(X,y,Y,B,g):
    roundEb = [0 for i in range(len(B))]
    for i in range(len(B)):
        for j in range(len(X)):
            roundEb[i] += -(Y[j]-y[j])*X[j][i]
    for i in range(len(B)):
        B[i] -=  g*roundEb[i]
    return B


#--------------------------------------------------------------



#CSV read (We don't use the coordinate data)
with open ("forestfires.csv","r" ) as f:
    reader = csv.reader(f)
    header = next(reader)
    for line in reader:
        X.append([1,line[2],line[3],float(line[4]),float(line[5]),float(line[6]),float(line[7]),float(line[8]),float(line[9]),float(line[10]),float(line[11])])
        Y_area.append(float(line[12]))
        
#Data processing and preliminary arrangements
#Change to log scale(to improve symmetry to improve regression results for right-skewed targets)
for i in range(len(Y_area)):
    Y_area[i] = math.log(Y_area[i]+1)
# plt.hist(Y_area)

#Change str(X[i][3],X[i][4]) to a discrete number
for i in range(len(X)):
    if X[i][1] == "jan":
        X[i][1] = 1
    elif X[i][1] == "feb":
        X[i][1] = 2
    elif X[i][1] == "mar":
        X[i][1] = 3
    elif X[i][1] == "apr":
        X[i][1] = 4
    elif X[i][1] == "may":
        X[i][1] = 5
    elif X[i][1] == "jun":
        X[i][1] = 6
    elif X[i][1] == "jul":
        X[i][1] = 7
    elif X[i][1] == "aug":
        X[i][1] = 8
    elif X[i][1] == "sep":
        X[i][1] = 9
    elif X[i][1] == "oct":
        X[i][1] = 10
    elif X[i][1] == "nov":
        X[i][1] = 11
    elif X[i][1] == "dec":
        X[i][1] = 12 
        
    if X[i][2] == "sun":
        X[i][2] = 1
    elif X[i][2] == "mon":
        X[i][2] = 2
    elif X[i][2] == "tue":
        X[i][2] = 3
    elif X[i][2] == "wed":
        X[i][2] = 4
    elif X[i][2] == "thu":
        X[i][2] = 5
    elif X[i][2] == "fri":
        X[i][2] = 6
    elif X[i][2] == "sat":
        X[i][2] = 7

#prepare for the ceoss varidation
X_train = X[:300]
X_test = X[300:]
Y_train = Y_area[:300]
Y_test = Y_area[300:]

#prepare some matrixes
B = [0.02 for i in range(len(X[0]))]
series_E = []


#--------------------------------------------------------------


#Body
#Training
y = predict(X_train,B)
print("Initial Loss = ",Loss(y,Y_train))

for i in range(2000):
    B = GradientDescent(X_train,y,Y_train,B,10**(-8))
    y = predict(X_train,B)
    series_E.append(Loss(y,Y_train))
#     print(Loss(y,Y_area))
Final_B = B
print("Final Loss = ",Loss(y,Y_train)) 
# print("Parameters B = ",B)


#Test
y_test = predict(X_test,Final_B)
print("Loss_test = ",Loss(y_test,Y_test))
#--------------------------------------------------------------


#Plot
t = [i for i in range(len(series_E))]
plt.plot(t,series_E)
plt.show


