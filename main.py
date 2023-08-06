import Lyu2023 as lyu
import newmodel as nm
import Cortes2010 as ct
import Rais2014 as rs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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
        results.append(ct.cortesModel(file) + rs.raisModel(file))
        
    csvIndex = ['Instace name', 'Cortes\'s Obj.Value', 'Cortes\'s t(s)', 'Rais\'s Obj.Value', 'Rais\'s t(s)']
    resultDf = pd.DataFrame(results, columns = csvIndex)
    resultDf.to_csv("result_Cortes_Rais_PDPT.csv", encoding='utf-8')

main()
