#!/usr/bin/env python
import numpy as np
import random
import os

if __name__ == '__main__':
    current_directory = os.getcwd()

    executionLine = 'python /path_where_runFwFM_is/runFwFM.py'

    rgIndStep, rgEmbSz, rgMask, rgScale, rgLoss = 5, 3, 2, 2, 2

    allLinesToexecute = [executionLine  + '  '  + str(i1)  + ' '  + str(i2)  + ' '  + str(i3)  + ' '  + \
        str(i4)  + ' '  + str(i5)   for i1 in range(rgIndStep)
                                    for i2 in range(rgEmbSz) 
                                    for i3 in range(rgMask) 
                                    for i4 in range(rgScale)
                                    for i5 in range(rgLoss)]

    count = 0
    np.random.seed(12890)
    numOfExLines = len(allLinesToexecute)
    random.shuffle(allLinesToexecute)

    execPerFile = 15
    while count*execPerFile <numOfExLines:
        with open(current_directory  + '/task_nn_'  + str(count), 'w') as file_task:
            for i in range(count*execPerFile, min((count + 1)*execPerFile, numOfExLines)):
                file_task.write(allLinesToexecute[i] + '\n')
        count  += 1