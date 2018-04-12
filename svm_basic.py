from classifier import classifier
from optStructK import optStructK
from numpy import *

class svm_basic:

    def __init__(self):
        self.b = None
        self.alphas = None
        self.weights = None

    def calcEkK(self, oS, k):
        fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
        Ek = fXk - float(oS.labelMat[k])
        return Ek

    def selectJrand(self, i,m):
        j=i #we want to select any J not equal to i
        while (j==i):
            j = int(random.uniform(0,m))
        return j

    def clipAlpha(self, aj,H,L):
        if aj > H: 
            aj = H
        if L > aj:
            aj = L
        return aj

    def selectJK(self, i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
        maxK = -1; maxDeltaE = 0; Ej = 0
        oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
        validEcacheList = nonzero(oS.eCache[:,0].A)[0]
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
                if k == i: continue #don't calc for i, waste of time
                Ek = self.calcEkK(oS, k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k; maxDeltaE = deltaE; Ej = Ek
            return maxK, Ej
        else:   #in this case (first time around) we don't have any valid eCache values
            j = self.selectJrand(i, oS.m)
            Ej = self.calcEkK(oS, j)
        return j, Ej

    def updateEkK(self, oS, k):#after any alpha has changed update the new value in the cache
        Ek = self.calcEkK(oS, k)
        oS.eCache[k] = [1,Ek]

    def innerLK(self, i, oS):
        Ei = self.calcEkK(oS, i)
        if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
            j,Ej = self.selectJK(i, oS, Ei) #this has been changed from selectJrand
            alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
            if (oS.labelMat[i] != oS.labelMat[j]):
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L==H: print("L==H"); return 0
            eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
            if eta >= 0: print("eta>=0"); return 0
            oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
            oS.alphas[j] = self.clipAlpha(oS.alphas[j],H,L)
            self.updateEkK(oS, j) #added this for the Ecache
            if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
            oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
            self.updateEkK(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
            b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
            b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
            if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
            elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
            else: oS.b = (b1 + b2)/2.0
            return 1
        else: return 0

    def smoPK(self, dataMatIn, classLabels, C, toler, maxIter):
        oS = optStructK(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
        iter = 0
        entireSet = True; alphaPairsChanged = 0
        while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
            alphaPairsChanged = 0
            if entireSet:   #go over all
                for i in range(oS.m):        
                    alphaPairsChanged += self.innerLK(i,oS)
                    print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
                iter += 1
            else: #go over non-bound (railed) alphas
                nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerLK(i,oS)
                    print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
                iter += 1
            if entireSet: entireSet = False #toggle entire set loop
            elif (alphaPairsChanged == 0): entireSet = True  
            print("iteration number: %d" % iter)
        return oS.b,oS.alphas

    def calcWs(self, alphas,dataArr,classLabels):
        X = mat(dataArr); labelMat = mat(classLabels).transpose()
        m,n = shape(X)
        w = zeros((n,1))
        for i in range(m):
            w += multiply(alphas[i]*labelMat[i],X[i,:].T)
        return w

    def fit(self, X, Y):
        self.b, self.alphas = self.smoPK(X, Y, 200, 0.0001, 10000)
        ws = self.calcWs(self.alphas, X, Y)
        self.weights = [self.b.getA()[0][0], ws[0][0], ws[1][0]]
        return self.weights

    def predict(self, X):
        hypotheses = []
        for x in X:
            result = self.weights[0] + self.weights[1] * x[1] + self.weights[2] * x[2]
            if result > 0:
                hypotheses.append(1)
            else:
                hypotheses.append(-1)
        return hypotheses