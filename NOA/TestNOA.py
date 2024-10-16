from NOA import NOA
import numpy as np
from DrawSensor import show_sensor_matrix
from PIL import Image
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle
import csv

nPop = 50  # Number of search agents
MaxIt = 2000  # Maximum number of function evaluations
nNode = 40
Rs = 10
Rc = 10
VarMin = 0
VarMax = 100

image = Image.open('C1_v2.png')
image_resized = image.resize((VarMax+1, VarMax+1))
image_L = image_resized.convert('L')
Area1 = np.zeros((VarMax+1,VarMax+1))
image_matrix = np.array(image_L)

image_1 =  Image.open('C1_real.png')

for i in range(VarMax+1):
    for j in range(VarMax+1):
        if image_matrix[i,j]  > 1:
            Area1[i,j] = 255
        else:
            Area1[i,j] = 1

ban_position_list = np.argwhere(Area1 == 1)
ban_position = [[x, y] for x, y in ban_position_list]

# Call the NOA optimization algorithm (Nutcracker Optimizer)
best_score, bestSol, convergence_curve, t = NOA(
    nPop, MaxIt,
    nNode, Rs, Rc,
    VarMin, VarMax,
    ban_position, Area1
)

bestSol = bestSol.reshape(-1, 2)

print(f"Best Fitness: {1 - best_score:.4f}")

with open('FOA.CSV', mode ='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([1 - best_score])
    csv_writer.writerows(bestSol)

show_sensor_matrix(nNode, Rs, bestSol, (1 -best_score) * 100, ban_position)

