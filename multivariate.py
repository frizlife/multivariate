import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython.core.display import HTML
import matplotlib.pyplot as plt

loansData = pd.read_csv('C:\Users\lfabrizio\AppData\Local\Continuum\Anaconda\Thinkful\Projects\multivariate_analysis\loansData.csv', index_col=0)
k = loansData['Monthly_Income']*12
loansData['Monthly_Income'] = k
X = loansData['Monthly_Income']
y = loansData['Interest_Rate'].map(lambda x: round(float(x.rstrip('%'))/100, 4))
loansData[['Interest_Rate']]= y
print(loansData.head())
print(X)
print(y)


#1. fit a OLS model with intercept on Annual Income
X = sm.add_constant(X)

est = sm.OLS(y, X, missing='drop').fit()
print("Monthly Income x Interest Rate")
print("")
print (est.summary())

#2. add home ownership
est = smf.ols(formula = 'Interest_Rate ~ Monthly_Income + C(Home_Ownership)', data = loansData).fit()
print("Adding Home Ownership")
print("")
print(est.summary())

#3. add interaction
income_linspace = np.linspace(loansData.Monthly_Income.min(), loansData.Monthly_Income.max(), 100)



est = smf.ols(formula = 'Interest_Rate ~ Monthly_Income * C(Home_Ownership)', data=loansData).fit()
print("With Interaction")
print("")
print(est.summary())

plt.scatter(loansData.Monthly_Income, loansData.Interest_Rate, alpha=.3)
plt.xlabel('Yearly Income')
plt.ylabel('Interest Rate')

#plt.plot(income_linspace, est.params[0] + est.params[1] * income_linspace *1 + est.params[2] * income_linspace * 0 + est.params[3] * income_linspace * 0 + est.params[4] * income_linspace * 0+ est.params[5] * income_linspace * 0+ est.params[6] * income_linspace * 0+ est.params[7] * income_linspace * 0+ est.params[8] * income_linspace * 0+ est.params[9] * income_linspace * 0, 'r')
#plt.plot(income_linspace, est.params[0] + est.params[1] * income_linspace *0 + est.params[2] * income_linspace * 1 + est.params[3] * income_linspace * 0 + est.params[4] * income_linspace * 0+ est.params[5] * income_linspace * 0+ est.params[6] * income_linspace * 0+ est.params[7] * income_linspace * 0+ est.params[8] * income_linspace * 0+ est.params[9] * income_linspace * 0, 'r')
#plt.plot(income_linspace, est.params[0] + est.params[1] * income_linspace *0 + est.params[2] * income_linspace * 0 + est.params[3] * income_linspace * 1 + est.params[4] * income_linspace * 0+ est.params[5] * income_linspace * 0+ est.params[6] * income_linspace * 0+ est.params[7] * income_linspace * 0+ est.params[8] * income_linspace * 0+ est.params[9] * income_linspace * 0, 'r')
#plt.plot(income_linspace, est.params[0] + est.params[1] * income_linspace *0 + est.params[2] * income_linspace * 0 + est.params[3] * income_linspace * 0 + est.params[4] * income_linspace * 1+ est.params[5] * income_linspace * 0+ est.params[6] * income_linspace * 0+ est.params[7] * income_linspace * 0+ est.params[8] * income_linspace * 0+ est.params[9] * income_linspace * 0, 'r')
plt.plot(income_linspace, est.params[0] + est.params[1] * income_linspace *0 + est.params[2] * income_linspace * 0 + est.params[3] * income_linspace * 0 + est.params[4] * income_linspace * 0+ est.params[5] * income_linspace * 1+ est.params[6] * income_linspace * 0+ est.params[7] * income_linspace * 0+ est.params[8] * income_linspace * 0+ est.params[9] * income_linspace * 0, 'b')
#plt.plot(income_linspace, est.params[0] + est.params[1] * income_linspace *0 + est.params[2] * income_linspace * 0 + est.params[3] * income_linspace * 0 + est.params[4] * income_linspace * 0+ est.params[5] * income_linspace * 0+ est.params[6] * income_linspace * 1+ est.params[7] * income_linspace * 0+ est.params[8] * income_linspace * 0+ est.params[9] * income_linspace * 0, 'r')
#plt.plot(income_linspace, est.params[0] + est.params[1] * income_linspace *0 + est.params[2] * income_linspace * 0 + est.params[3] * income_linspace * 0 + est.params[4] * income_linspace * 0+ est.params[5] * income_linspace * 0+ est.params[6] * income_linspace * 0+ est.params[7] * income_linspace * 1+ est.params[8] * income_linspace * 0+ est.params[9] * income_linspace * 0, 'r')
#plt.plot(income_linspace, est.params[0] + est.params[1] * income_linspace *0 + est.params[2] * income_linspace * 0 + est.params[3] * income_linspace * 0 + est.params[4] * income_linspace * 0+ est.params[5] * income_linspace * 0+ est.params[6] * income_linspace * 0+ est.params[7] * income_linspace * 0+ est.params[8] * income_linspace * 1+ est.params[9] * income_linspace * 0, 'r')
plt.plot(income_linspace, est.params[0] + est.params[1] * income_linspace *0 + est.params[2] * income_linspace * 0 + est.params[3] * income_linspace * 0 + est.params[4] * income_linspace * 0+ est.params[5] * income_linspace * 0+ est.params[6] * income_linspace * 0+ est.params[7] * income_linspace * 0+ est.params[8] * income_linspace * 0+ est.params[9] * income_linspace * 1, 'g')
plt.show()
