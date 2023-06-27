import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import gurobipy as gp
from gurobipy import GRB

filename = "./PDPTWT/3R4K4T/3R-4K-4T-180L-7.txt"

# Read the meta-data of problem (number of requests, number of vehicles, number of transport stations, capability of vehicles)
def readMetaData(filename):
    metaData = pd.read_csv(filename, nrows=2, sep= '\t', on_bad_lines='skip')
    return metaData

# Read the instance's data (name of node, location (x, y), time-windows, load of the request)
def readDataframe(filename):
    df = pd.read_csv(filename, skiprows=3, sep='\t')
    temp = df[df['node'].str.constains("t") == True]
    df = df.drop(df[df['node'].str.contains("t")].index)
    for index, row in temp.iterrows():
        if row['node'].str.containts("t"):
            df.add(row.replace("t", "ts"))
            df.add(row.replace("t", "tf"))
    df = df.reset_index(drop=True)
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
def calculateDistance(x1, x2, y1, y2):
    return math.sqrt((x2 - x1)**2 + (y2-y1)**2)

# one unit of distance can be traveled in one time unit
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
    transferStart = df.loc[df['node'].str.contains('ts'),'node']
    transferFinish = df.loc[df['node'].str.contains('tf'), 'node']
    return {"a":allNodes, "ro":rOrigins, "rd":rDestinations, "vo":vOrigins, "vd":vDestinations, "ts":transferStart, "tf":transferFinish}

# Model
def cortesModel(filename):
    model = gp.Model()
    metaData = readMetaData(filename)
    df = readDataframe(filename)
    nodeList = getNodeList(df)


    ##TODO: create E (set of considerable arcs) to check validity of the constraints
    arcs = []
    for i in nodeList['vo'].values:
        for j in nodeList['ro'].values:
            arcs.add((i,j))
        for j in nodeList['ts'].values:
            arcs.add((i,j))
        for j in nodeList['vd'].values:
            arcs.add((i, i.replace("o", "e")))

    for i in nodeList['ro'].values:
        for j in np.concatenate((nodeList['ro'].values, nodeList['rd'].values, nodeList['ts'].values)):
            if i != j:
                arcs.add((i,j))

    for i in nodeList['rd'].values:
        for j in np.concatenate((nodeList['ro'].values, nodeList['rd'].values, nodeList['vd'].values, nodeList['ts'].values)):
            if i != j or (j in nodeList['ro'].values and i != j.replace("p","d")):
                arcs.add((i,j))

    for i in nodeList['ts'].values:
        for j in nodeList['tf'].values:
            if i == j.replace("f", "s"):
                arcs.add((i,j))

    for i in nodeList['tf'].values:
        for j in np.concatenate((nodeList['ro'].values, nodeList['rd'].values, nodeList['vo'].values, nodeList['ts'].values)):
            if i != j.replace("f") and j in nodeList['ts'].values:
                arcs.add((i,j))

    # Testing Symmetries Breaking Constraints
    # df.loc[df['node'].str.contains('o'), 'x'] = 50
    # df.loc[df['node'].str.contains('o'), 'y'] = 50
    # df.loc[df['node'].str.contains('e'), 'x'] = 50
    # df.loc[df['node'].str.contains('e'), 'y'] = 50

    nRequests = int(metaData['nr'])
    nVehicles = int(metaData['nv'])
    nTransports = int(metaData['nt'])
    vCapability = int(metaData['capacity'])

    c = pd.DataFrame.from_dict(distancesMatrix(df)).fillna(0)
    # k = pd.RangeIndex(nVehicles)
    # r = pd.RangeIndex(nRequests)
    # u = pd.Series(index=k, data=np.full(nVehicles, vCapability))
    # q = pd.Series(index = np.concatenate((nodeList['ro'].values, nodeList['rd'].values)), data=df.loc[0:nRequests*2-1,'load'].values, dtype=int)

    cost = {(i, j): c.loc[i, j].values(0) for i in c.index for j in c.column}

    print(df)

    xIndex = [(k, i, j) for k in pd.RangeIndex(nVehicles) for (i,j) in arcs]
    zIndex = [(k, i, j) for k in pd.RangeIndex(nVehicles) for i in pd.RangeIndex(nRequests) for j in nodeList['a'].values]
    aIndex = [(k, i) for k in pd.RangeIndex(nVehicles) for i in nodeList['a'].values]
    bIndex = [(k, i) for k in pd.RangeIndex(nVehicles) for i in nodeList['a'].values]

    x = model.addVars(xIndex, vtype=GRB.BINARY, name='x')
    a = model.addVars(aIndex, vtype=GRB.INTEGER, name='a')
    b = model.addVars(bIndex, vtype=GRB.INTEGER, name='b')

    model.setObjective(sum((c[i][j] * x[k, i, j]) for (i,j) in arcs for k in pd.RangeIndex(nVehicles)), GRB.MINIMIZE)
    model.update()

    ## Constaints                  
    




    model.update()
    model.optimize()
    # model.computeIIS()
    # model.write("model.ilp")
    
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
                plt.plot([x1, x2], [y1, y2], 'gray', linestyle="--")
        plt.show()
    
    #plotLocation(df)
    infos = [filename, model.getObjective().getValue(), model.Runtime]
    return infos