import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
import seaborn as sns
import requests
import io

class Analyzer():

    def __init__(self, filePath, outputName):
        self.filePath = filePath
        self.outputName = outputName
        self.data = pd.read_csv(self.filePath)
        # FIPS Codes - IDGAF
        fipsCodes="https://www2.census.gov/geo/docs/reference/state.txt"
        fips=requests.get(fipsCodes).content
        fips = pd.read_table(io.StringIO(fips.decode('utf-8')), sep='|') 
        fips.drop('STATENS', axis=1, inplace=True)
        # Add State Names to Data
        self.data = self.data.merge(fips, how='left', left_on='FIPSStateCode', right_on='STATE')
        # convoluted af
        self.data.drop(['STATE', 'STUSAB','FIPSStateCode'] ,axis=1, inplace=True)
        self.data['FIPSStateCode'] = self.data['STATE_NAME']
        self.data.drop('STATE_NAME' ,axis=1, inplace=True)
    def clean(self):
        data = self.data
        '''
        bankMapper = {'Atlanta':1, 'Boston':2,
                  'Chicago':3, 'Cincinnati':4,
                  'Dallas':5, 'Des Moines':6,
                  'Indianapolis':7, 'New York':8,
                   'Pittsburgh':9, 'San Francisco':10,
                   'Topeka':11,}
        '''
        stateMapper = {
"Maryland":75847,
"Hawaii":73486,
"Alaska":73355,
"New Jersey":72222,
"Connecticut":71346,
"Massachusetts":70628,
"New Hampshire":70303,
"Virginia":66262,
"California":64500,
"Washington":64129,
"Colorado":63909,
"Minnesota":63488,
"Utah":62912,
"Delaware":61255,
"New York":60850,
"North Dakota":60557,
"Wyoming":60214,
"Illinois":59588,
"Rhode Island":58073,
"Vermont":56990,
"Pennsylvania":55702,
"Texas":55653,
"Wisconsin":55638,
"Nebraska":54996,
"Iowa":54736,
"Oregon":54148,
"Kansas":53906,
"South Dakota":53017,
"Nevada":52431,
"Maine":51494,
"Arizona":51492,
"Georgia":51244,
"Michigan":51084,
"Ohio":51075,
"Indiana":50532,
"Missouri":50238,
"Montana":49509,
"Florida":49426,
"Oklahoma":48568,
"Idaho":48275,
"North Carolina":47830,
"Tennessee":47275,
"South Carolina":47238,
"Louisiana":45727,
"New Mexico":45382,
"Kentucky":45215,
"Alabama":44765,
"West Virginia":42019,
"Arkansas":41995,
"Mississippi":40593,
"District of Columbia":75628,
"Guam":73298,
}
        
        data['FIPSStateCode'].replace(stateMapper, inplace=True)
        
        bankMapper = {'Atlanta':472522, 'Boston':673184,
                  'Chicago':2705000, 'Cincinnati':298800,
                  'Dallas':1318000, 'Des Moines':215472,
                  'Indianapolis':864771, 'New York':8538000,
                   'Pittsburgh':303625, 'San Francisco':864816,
                   'Topeka':126808,}
        
        data['FHLBank'].replace(bankMapper, inplace=True)
        
        propMapper = {'PT01':1, 'PT02':2, 'PT03':3, 'PT04':4, 'PT05':5,
                      'PT06':6, 'PT07':7, 'PT08':8, 'PT09':9, 'PT10':10,
                      'PT11':11, 'PT12':12}
        
        data['PropType'].replace(propMapper, inplace=True)
        self.allColumns = ['FHLBank', 'FIPSStateCode', 'FIPSCountyCode', 'Income', 'Purpose', 'Term', 'First', 
                    'BoAge', 'Occup', 'Rate', 'BoCreditScore', 'PropType', 'Amount',]
        data = data[self.allColumns]
        
        # These variables are not well correlated with the target
        self.dropperz = ['Occup', 'FIPSCountyCode', 'PropType',]
        data.drop(self.dropperz, axis=1, inplace=True)
        self.corrs = data.corr()
        #data.to_csv('clean.csv', index=False)
        data = data.as_matrix()
        self.data = data        



    def splitData(self):

        data = self.data

        np.random.shuffle(data)
        #print(data[:10,:])

        t = int(data.shape[0] * .80)
        train = data[:t,:]
        test = data[t:,:]
        self.train = train
        self.test = test
    #########################
    ### Linear Regression ###
    #########################
    def runLinearReg(self):
        lm = LinearRegression()
        X = self.train[:,:-1]
        y = self.train[:,-1:]
        
        lm.fit(X, y)
        scores = lm.score(X, y)
        self.rSquared = scores
        # This is lazy 
        self.erTest = self.test.copy()
        self.rrTest = self.test.copy()
        
        X = self.test[:,:-1]
        preds = lm.predict(X)
                
        results = pd.DataFrame(self.test)
        self.intercept = lm.intercept_
        self.coef = lm.coef_
        cols = self.allColumns
        for drop in self.dropperz:
            if drop in cols: 
                cols.remove(drop)
        results['rr_Amount'] = preds

        cols.append('rr_Amount')
        results.columns = cols
        self.modColumns = cols

        
        self.test = results
        yTrue = self.test.loc[:,'Amount']
        yPred = self.test.loc[:,'rr_Amount']
        
        self.mse = sklearn.metrics.mean_squared_error(yTrue, yPred)
        self.mae = sklearn.metrics.mean_absolute_error(yTrue, yPred)

        self.r2_score = sklearn.metrics.r2_score(yTrue, yPred)

        self.lm = lm

    #############################
    ### ElasticNet Regression ###
    #############################
    def elasticRegression(self):
        er = ElasticNet(alpha=0.0)
        X = self.train[:,:-1]
        y = self.train[:,-1:]
        
        er.fit(X, y)
        self.erScore = er.score(X, y)

        X = self.erTest[:,:-1]
        
        preds = er.predict(X)

        self.erTest = pd.DataFrame(self.erTest)
        self.erTest['er_Amount'] = preds
        
        self.erTest.to_csv('Elastic_Net_Apply.csv')
        
        yTrue = self.erTest.loc[:,9]
        yPred = self.erTest.loc[:,'er_Amount']
        
        self.erMse = sklearn.metrics.mean_squared_error(yTrue, yPred)
        self.erMae = sklearn.metrics.mean_absolute_error(yTrue, yPred)

        self.erCoef = er.coef_
        self.erNIter = er.n_iter_
        
        self.er = er

    ########################
    ### Ridge Regression ###
    ########################
    def ridgeRegression(self):
        rr = Ridge()
        X = self.train[:,:-1]
        y = self.train[:,-1:]
        
        rr.fit(X, y)
        self.rrScore = rr.score(X, y)

        X = self.rrTest[:,:-1]        
        preds = rr.predict(X)

        self.rrTest = pd.DataFrame(self.rrTest)
        self.rrTest['er_Amount'] = preds

        yTrue = self.rrTest.loc[:,9]
        yPred = self.rrTest.loc[:,'er_Amount']
        
        self.rrMse = sklearn.metrics.mean_squared_error(yTrue, yPred)
        self.rrMae = sklearn.metrics.mean_absolute_error(yTrue, yPred)
    
        self.rrCoef = rr.coef_
        self.rr = rr
       
        
    '''
    def plotPred(self):
        toPlot = [#'FHLBank',
                  #'FIPSStateCode',
                  #'FIPSCountyCode',
                  'Income',
                  'Purpose',
                  'Term',
                  'First',
                  #'BoAge',
                  #'Rate',
                  'BoCreditScore',
                  'PropType',]
                  #'Amount',] 
        plt.scatter(self.test.loc[:, 'rr_Amount'], self.test['Amount'],  color='black')
        plt.plot(self.test.loc[:, toPlot], self.test['rr_Amount'], color='blue', linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.show()
    '''

    def makeCsv(self, type='test'):
        if type.lower() == 'test':
            self.test.to_csv('test_apply.csv', index=False)
        
    def visualize(self):
        sns.pairplot(self.test, x_vars=['Income','Purpose','Term', 'First', 'BoCreditScore', 'PropType'], y_vars='rr_Amount', size=7, aspect=0.7)
        plt.show()

    def visLeastSquares(self):
        sns.pairplot(self.test, x_vars=['Income', 'Term', 'BoCreditScore',], y_vars='Amount', size=7, aspect=0.7, kind='reg')
        plt.show()
    #sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7, kind='reg')

    def simulate(self, values):
        results = {'Linear':0.0, 'Elastic':0.0, 'Ridge':0.0,}
        apply = np.array(values)
        apply = apply.reshape(1, -1) 
        p = self.lm.predict(apply)
        results['Linear'] = p
        p = self.er.predict(apply)
        results['Elastic'] = p
        p = self.rr.predict(apply)
        results['Ridge'] = p
        
        return results
        
 
analyzer = Analyzer('ChosenColumns.csv', 'CleanFHA.csv')
analyzer.clean()
analyzer.splitData()
analyzer.runLinearReg()
analyzer.elasticRegression()
analyzer.ridgeRegression()

print('#############################')
print('### Non-Machine Learning  ###')
print('#############################')
print()
print('Correlations To Amount')
print(analyzer.corrs.loc[:, 'Amount'])
print()
print('#########################')
print('### Linear Regression ###')
print('#########################')
print()
print('R2')
print(analyzer.rSquared)
print('Mean-Squared-Error')
print(analyzer.mse)
print('Mean-Absolute-Error')
print(analyzer.mae)
print()
print('Model Intercept')
print(analyzer.intercept) 
print()
print('Model Coefficients')
for i, coef in enumerate(analyzer.coef[0]):
    print(analyzer.modColumns[i] + ':   ' + str(coef))
print()
print('Variance Score')
print(analyzer.r2_score)

print('#############################')
print('### ElasticNet Regression ###')
print('#############################')
print()
print('R2')
print(analyzer.erScore)
print()
print('Model Coefficients')
for i, coef2 in enumerate(analyzer.erCoef):
    print(analyzer.modColumns[i] + ':   ' + str(coef2))
print()
print('Number of Iterations')
print(analyzer.erNIter)
print()
print()
print('Mean-Squared-Error')
print(analyzer.erMse)
print('Mean-Absolute-Error')
print(analyzer.erMae)

print('########################')
print('### Ridge Regression ###')
print('########################')
print()
print('R2')
print(analyzer.rrScore)
print()
print('Model Coefficients')
for i, coef3 in enumerate(analyzer.rrCoef[0]):
    print(analyzer.modColumns[i] + ':   ' + str(coef3))
print()
print('Mean-Squared-Error')
print(analyzer.rrMse)
print('Mean-Absolute-Error')
print(analyzer.rrMae)
print()

maxer = [analyzer.rSquared, analyzer.erScore, analyzer.rrScore]
maxer = max(maxer)
if analyzer.rSquared == maxer:
    print('R2 - Linear Reg Wins')
elif analyzer.erScore == maxer:
    print('R2 - Elastic Net Wins')
elif analyzer.rrScore == maxer:
    print('R2 - Ridge Regression Wins')

miner = [analyzer.rrMae, analyzer.erMae, analyzer.mae]
miner = min(miner)
if analyzer.mae == miner:
    print('Mae - Linear Reg Wins')
elif analyzer.erMae == miner:
    print('Mae - Elastic Net Wins')
elif analyzer.rrMae == miner:
    print('Mae - Ridge Regression Wins')

print(analyzer.simulate([
# FHLBank
36473,
# FIPSStateCode
56000,
# Income
30000,
# Purpose
1,
# Term
360,
# First
1,
# BoAge
45,
# Rate
0.0363,
# BoCreditScore
5,
]))


