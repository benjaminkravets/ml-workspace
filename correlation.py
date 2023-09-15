# Creating a sample data frame
import pandas as pd
ColumnNames=['CIBIL','AGE','GENDER' ,'SALARY', 'APPROVE_LOAN']
DataValues=[ [480, 28, 'M', 610000, 'Yes'],
             [480, 42, 'M',140000, 'No'],
             [480, 29, 'F',420000, 'No'],
             [490, 30, 'M',420000, 'No'],
             [500, 27, 'M',420000, 'No'],
             [510, 34, 'F',190000, 'No'],
             [550, 24, 'M',330000, 'Yes'],
             [560, 34, 'M',160000, 'Yes'],
             [560, 25, 'F',300000, 'Yes'],
             [570, 34, 'M',450000, 'Yes'],
             [590, 30, 'F',140000, 'Yes'],
             [600, 33, 'M',600000, 'Yes'],
             [600, 22, 'M',400000, 'Yes'],
             [600, 25, 'F',490000, 'Yes'],
             [610, 32, 'M',120000, 'Yes'],
             [630, 29, 'F',360000, 'Yes'],
             [630, 30, 'M',480000, 'Yes'],
             [660, 29, 'F',460000, 'Yes'],
             [700, 32, 'M',470000, 'Yes'],
             [740, 28, 'M',400000, 'Yes']]
 
#Create the Data Frame
LoanData=pd.DataFrame(data=DataValues,columns=ColumnNames)
print(LoanData.head())
#########################################################
# Cross tabulation between GENDER and APPROVE_LOAN
CrosstabResult=pd.crosstab(index=LoanData['GENDER'],columns=LoanData['APPROVE_LOAN'])
print(CrosstabResult)

# importing the required function
from scipy.stats import chi2_contingency

# Performing Chi-sq test
ChiSqResult = chi2_contingency(CrosstabResult)

# P-Value is the Probability of H0 being True
# If P-Value&gt;0.05 then only we Accept the assumption(H0)

print('The P-Value of the ChiSq Test is:', ChiSqResult[1])