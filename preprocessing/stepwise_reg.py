import numpy
from sklearn import linear_model
import matplotlib.pyplot as plot
import csv
import os

from sklearn.metrics import r2_score
import statsmodels.api as sm
_author_ = 'Sebastian Palacio'

class stepwise_regression:


    def setpwise_reg(xList, target, names):

        def xattrSelect(x,idxSet):  # Toma la matrix X de la regresión y = betaX y la convierte en una lista de listas y retorna un subconjunto que contiene columnass en idxSet
            xOut = []
            for row in x:
                xOut.append([row[i] for i in idxSet])
            return (xOut)


        # dividimos los atributos y el target en training y test

        # armamos columnas de atributos para ir probando que columnas quedan mejor
        attributeList = []
        index = range(len(xList[1]))  # toma la cantidad de columnas
        indexSet = set(
            index)  # crea una variable ordenada en función de los valores que tenga la variable de la cual se origina
        indexSeq = []
        oosError = []
        aasError = []
        betas = []
        bbs = []
        sse = []
        for i in index:
            attSet = set(attributeList)  # marca las columnas que se van descartando

            attTrySet = indexSet - attSet  # marca las columnas que se van utilizando (el contrario al attSet)

            attTry = [ii for ii in attTrySet]  # paso attTrySet a una lista [0.0, 1.0...31]

            errorList = []
            absError = []
            attTemp = []
            betaList = []
            seList = []
            ####
            for iTry in attTry:
                attTemp = [] + attributeList  # comienza como una columna de [1...31], luego itera nuevamente con dos columnas [31,1..31], luego con tres [31,21,1...31] siempre iterando la ultima columna (esto lo hace usando las columnas que genera attSet)
                attTemp.append(iTry)

                ####
                xTrainTemp = xattrSelect(xList, attTemp)
                ####
                xTrain = numpy.array(xTrainTemp)
                yTrain = numpy.array(target)

                fileQModel = sm.OLS(endog=yTrain, exog=xTrain, missing='none')
                result = fileQModel.fit()

                rmsError = result.aic  # este va calculando todo el tiempo el RMS
                errorList.append(rmsError)  # arma una columna de los RMS
                betaList.append(betas)

            iBest = numpy.argmin(errorList)  # toma el mínimo RMS
            attributeList.append(attTry[iBest])  # toma la mejor combinación de columnas de donde salió el mínimo RMS
            oosError.append(errorList[iBest])  # va armando una columna de mejores RMS
            bbs.append(betaList[iBest])



        print('Out of Sample error versus attribute set Size ', oosError)
        indexBest = oosError.index(min(oosError))
        print('index Best (min oosError = ', indexBest)
        print('Out of Sample absolute error versus attribute set Size ', aasError)
        print('\n Best attribute indices ', attributeList)
        namesList = [names[i] for i in attributeList]
        namesList = namesList[:indexBest+1]
        print('\n Best Attribute Names ', namesList)
        print('betas',bbs[indexBest])
        # numpy.savetxt("results\stepwise\oosError.csv", oosError, delimiter=",")
        # numpy.savetxt("results\stepwise\\attributeList.csv", attributeList, delimiter=",")






        '''
        Esto arroja no solo el nombre de los atributos, sino su orden de calidad en términos de predicción. !!!!!
        '''

        # PLOT ERROR VERSUS NUMBER OF ATTRIBUTES#####################################################################################
        x = range(len(oosError))

        plot.plot(x, oosError,
                  'k')  # plot(x,y,keyword arguments) if i put 'k' is black color lines colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
        plot.xlabel('Attributes')
        plot.ylabel('R2')
        plot.title('Stepwise Regression')
        # plot.show()


        return namesList
