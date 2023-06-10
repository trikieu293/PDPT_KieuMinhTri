# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import gurobipy as gp
from gurobipy import GRB

# %%
filename = "./PDPT-R5-K2-T1-Q100-6.txt"

# %%
# Read the meta-data of problem (number of requests, number of vehicles, number of transport stations, capability of vehicles)
def readMetaData(filename):
    metaData = pd.read_csv(filename, nrows=2, sep= '\t', on_bad_lines='skip')
    return metaData

# Read the instance's data (name of node, location (x, y), time-windows, load of the request)
def readDataframe(filename):
    df = pd.read_csv(filename, skiprows=3, sep='\t')
    return df

# Calculate the Euclid-distance between locations
def calculateDistance(x1, x2, y1, y2):
    return math.sqrt((x2 - x1)**2 + (y2-y1)**2)

# Generate distance-matrix between locations
def distancesMatrix(df):
    matrix = {}
    for location1 in df["node"]:
        for location2 in df["node"]:
            if location1 != location2:
                x1 = df.loc[df["node"]==location1, 'x']
                x2 = df.loc[df["node"]==location2, 'x']
                y1 = df.loc[df["node"]==location1, 'y']
                y2 = df.loc[df["node"]==location2, 'y']
                matrix[location1, location2] = calculateDistance(int(x1), int(x2), int(y1), int(y2))
    return matrix

# Generate dictionary of (node:load)
def loadDict(df):
    matrix = {}
    for location in df["node"]:
        matrix[location] = df.loc[df["node"]==location, 'load'].values[0]
    return matrix

# Get the list of grouped nodes
def getNodeList(df):
    rOrigins = df.loc[df['node'].str.contains('p'),'node']
    rDestinations = df.loc[df['node'].str.contains('d'),'node']
    vOrigins = df.loc[df['node'].str.contains('o'),'node']
    vDestinations = df.loc[df['node'].str.contains('e'),'node']
    transferNodes = df.loc[df['node'].str.contains('t'),'node']
    return [rOrigins, rDestinations, vOrigins, vDestinations, transferNodes]
def calculateDistance(x1, x2, y1, y2):
    return math.sqrt((x2 - x1)**2 + (y2-y1)**2)

def distancesMatrix(df):
    matrix = {}
    for location1 in df["node"]:
        matrix[location1] = {}
        for location2 in df["node"]:
            if location1 != location2:
                x1 = df.loc[df["node"]==location1, 'x'].values[0]
                x2 = df.loc[df["node"]==location2, 'x'].values[0]
                y1 = df.loc[df["node"]==location1, 'y'].values[0]
                y2 = df.loc[df["node"]==location2, 'y'].values[0]
                matrix[location1][location2] = calculateDistance(int(x1), int(x2), int(y1), int(y2))
    return matrix

def loadMatrix(df):
    matrix = {}
    for location in df["node"]:
        matrix[location] = df.loc[df["node"]==location, 'load'].values[0]
    return matrix

def getNodeList(df):
    allNodes = df['node']
    rOrigins = df.loc[df['node'].str.contains('p'),'node']
    rDestinations = df.loc[df['node'].str.contains('d'),'node']
    vOrigins = df.loc[df['node'].str.contains('o'),'node']
    vDestinations = df.loc[df['node'].str.contains('e'),'node']
    transferNodes = df.loc[df['node'].str.contains('t'),'node']
    return {"a":allNodes, "ro":rOrigins, "rd":rDestinations, "vo":vOrigins, "vd":vDestinations, "t":transferNodes}


# %%
# Model, Parameters, Variables and Objective Function
model = gp.Model()
metaData = readMetaData(filename)
df = readDataframe(filename)
nodeList = getNodeList(df)

nRequests = int(metaData['nr'])
nVehicles = int(metaData['nv'])
nTransports = int(metaData['nt'])
vCapability = int(metaData['capacity'])

c = pd.DataFrame.from_dict(distancesMatrix(df)).fillna(0)
k = pd.RangeIndex(nVehicles)
r = pd.RangeIndex(nRequests)
u = pd.Series(index=k, data=np.full(nVehicles, vCapability))
q = pd.Series(index = np.concatenate((nodeList['ro'].values, nodeList['rd'].values)), data=df.loc[0:nRequests*2-1,'load'].values, dtype=int)
M = 999

xIndex = [(k, i, j) for k in pd.RangeIndex(nVehicles) for i in nodeList['a'].values for j in nodeList['a'].values if i != j]
yIndex = [(k, r, i, j) for k in pd.RangeIndex(nVehicles) for r in pd.RangeIndex(nRequests) for i in nodeList['a'].values for j in nodeList['a'].values if i != j]
zIndex = [(k, i, j) for k in pd.RangeIndex(nVehicles) for i in nodeList['a'].values for j in nodeList['a'].values if i != j]
eIndex = [(k, i) for k in pd.RangeIndex(nVehicles) for i in nodeList['a'].values]
sIndex = [(k1, k2, t, r) for k1 in pd.RangeIndex(nVehicles) for k2 in pd.RangeIndex(nVehicles) for t in nodeList['t'].values for r in pd.RangeIndex(nRequests) if k1 != k2]

x = model.addVars(xIndex, vtype=GRB.BINARY, name='x')
y = model.addVars(yIndex, vtype=GRB.BINARY, name='y')
z = model.addVars(zIndex, vtype=GRB.BINARY, name='z')
e = model.addVars(eIndex, vtype=GRB.INTEGER, name='e')
s = model.addVars(sIndex, vtype=GRB.BINARY, name='s')

model.setObjective(sum((c[i][j] * x[k, i, j]) for i in nodeList['a'].values for j in nodeList['a'].values for k in pd.RangeIndex(nVehicles) if i != j), GRB.MINIMIZE)


# %%
# Constraints
for k in pd.RangeIndex(nVehicles):
    for i in nodeList['vo'].values:
        if str(k) in i:
            model.addConstr(sum(x[k, i, j] for j in nodeList['a'].values if j != i) == 1, name="constr25")
            for l in nodeList['vd'].values:
                if str(k) in l:
                    model.addConstr(sum(x[k, i, j] for j in nodeList['a'].values if j != i) == sum(x[k, j, l] for j in nodeList['a'].values if j != l), name="constr2")
    for i in nodeList['a']:
        if i not in [node for node in np.concatenate((nodeList['vo'].values, nodeList['vd'].values)) if str(k) in node]:
            model.addConstr(sum(x[k, i, j] for j in nodeList['a'].values if i != j) == sum(x[k, j, i] for j in nodeList['a'].values if i != j), name="constr3")

for r in pd.RangeIndex(nRequests):
    for i in nodeList['a'].values:
        if i in nodeList['ro'].values:
            if str(r) in i:
                model.addConstr(sum(sum(y[k, r, i, j] for j in nodeList['a'].values if i != j) for k in pd.RangeIndex(nVehicles)) == 1, name="constr4")
        if i in nodeList['rd'].values:
            if str(r) in i:
                model.addConstr(sum(sum(y[k, r, j, i] for j in nodeList['a'].values if i != j) for k in pd.RangeIndex(nVehicles)) == 1, name="constr5")
        if i in nodeList['t'].values:
            model.addConstr(sum(sum(y[k, r, i, j] for j in nodeList['a'].values if i != j) for k in pd.RangeIndex(nVehicles)) == sum(sum(y[k, r, j, i] for j in nodeList['a'].values if i != j) for k in pd.RangeIndex(nVehicles)), name="constr6")   
        for k in pd.RangeIndex(nVehicles):
            if i not in [node for node in nodeList['ro'].values if str(r) in node] and i not in [node for node in nodeList['rd'].values if str(r) in node] and i not in nodeList['t'].values:
                model.addConstr(sum(y[k, r, i, j] for j in nodeList['a'].values if i != j) == sum(y[k, r, j, i] for j in nodeList['a'].values if i != j), name="constr16")

for i in nodeList['a'].values:
    for j in nodeList['a'].values:
        for k in pd.RangeIndex(nVehicles):
            if i != j:
                for r in pd.RangeIndex(nRequests):
                    model.addConstr(y[k, r, i, j] <= x[k, i, j], name="constr8")
                model.addConstr(sum(q[r]*y[k, r, i, j] for r in pd.RangeIndex(nRequests)) <= u[k]*x[k, i, j], name="constr9")
                model.addConstr(x[k, i, j] <= z[k, i, j], name="constr17")
                model.addConstr(z[k, i, j] + z[k, j, i] == 1, name="constr18")
                for l in nodeList['a'].values:
                #if (i not in [node for node in nodeList['vo'].values if str(k) in node]) and (j not in [node for node in nodeList['vo'].values if str(k) in node]) and (l not in [node for node in nodeList['vd'].values if str(k) in node]):
                    if (j != l and i != l):
                        model.addConstr(z[k, i, j] + z[k, j, l] + z[k, l, i] <= 2, name="constr19")
                model.addConstr(e[k, i] + 1 - e[k, j] <= M*(1 - x[k, i, j]), name='constr20')
                model.addConstr(e[k, i] >= 0, name='constr23')

for r in pd.RangeIndex(nRequests):
    for t in nodeList['t'].values:
        for k1 in pd.RangeIndex(nVehicles):
            for k2 in pd.RangeIndex(nVehicles):
                if k1 != k2:
                    model.addConstr(sum(y[k1, r, j, t] for j in nodeList['a'].values if j != t) + sum(y[k2, r, t, j] for j in nodeList['a'].values if j != t) <= s[k1, k2, t, r] + 1, name='constr21')
                    model.addConstr(e[k1, t] - e[k2, t] <= M*(1 - s[k1, k2, t, r]), name='constr22')
                

# %%
model.optimize()

# %%
def plotLocation(df):
    fig, axes = plt.subplots(figsize=(10, 10))
    
    plt.scatter(df.loc[df['node'].str.contains('p'),'x'].values, df.loc[df['node'].str.contains('p'),'y'].values, s=50, facecolor='red', marker='o')
    plt.scatter(df.loc[df['node'].str.contains('d'),'x'].values, df.loc[df['node'].str.contains('d'),'y'].values, s=50, facecolor='green', marker='o')
    plt.scatter(df.loc[df['node'].str.contains('o'),'x'].values, df.loc[df['node'].str.contains('o'),'y'].values, s=50, facecolor='red', marker='s')
    plt.scatter(df.loc[df['node'].str.contains('e'),'x'].values, df.loc[df['node'].str.contains('e'),'y'].values, s=50, facecolor='green', marker='s')
    plt.scatter(df.loc[df['node'].str.contains('t'),'x'].values, df.loc[df['node'].str.contains('t'),'y'].values, s=50, facecolor='blue', marker='D')
    
    for xi, yi, text in zip(df['x'].values, df['y'].values, df['node'].values):
        plt.annotate(text, xy=(xi, yi), xycoords='data', xytext=(5, 5), textcoords='offset points')
    xResult = pd.DataFrame(x.keys(), columns=["k","i","j"])
    xResult["value"]=model.getAttr("X", x).values()
    
    for index, row in xResult.iterrows():
        if row["value"] == 1:
            x1 = df.loc[df['node'] == row["i"], 'x'].values
            y1 = df.loc[df['node'] == row["i"], 'y'].values
            x2 = df.loc[df['node'] == row["j"], 'x'].values
            y2 = df.loc[df['node'] == row["j"], 'y'].values
            plt.plot([x1, x2], [y1, y2])
    plt.show()
    
plotLocation(df)



