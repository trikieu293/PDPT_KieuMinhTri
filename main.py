import Lyu2023 as lyu
import newmodel as nm
import Cortes2010 as ct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import csv 
import gurobipy as gp
from gurobipy import GRB


def main():
    
    mainFolderPath = './PDPT/'
    folder = os.fsencode(mainFolderPath)
    filenames = []
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(('.txt')): 
            filenames.append(mainFolderPath + filename)
    
    results = []
    for file in filenames:
        print(file)
        results.append(lyu.lyuModel(file) + nm.newModel(file))
        
    csvIndex = ['Instace name', 'Lyu\'s Obj.Value', 'Lyu\'s t(s)', 'ValidConstr1\'s Obj.Value', 'ValidConstr1\'s t(s)']
    resultDf = pd.DataFrame(results, columns = csvIndex)
    resultDf.to_csv("result_SBC_normal.csv", encoding='utf-8')

main()
