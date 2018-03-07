#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 20:52:20 2018

@author: jiangyunzhang
"""

import pandas as pd
import numpy as np
import scipy.optimize as optimize
import math
import matplotlib.pyplot as plt

# Get the data from Excel sheets
Paydown = pd.read_csv('Paydown.csv')
Collateral = pd.read_csv('Collateral.csv')
Performance = pd.read_csv('Performance.csv')
STRIPS = pd.read_csv('US Sovereign Strips.csv')

# Q3 Cashflow to date
PSA = 171
Month = 179

# Get a function to calculate prepayment behavior based on PSA and number of month
def PSA_function(Month,PSA):
    X = pd.DataFrame(index = range(Month),columns=['CPR','SMM','FractionRemaining'])

    X['CPR'][0] = 0.002 * PSA / 100

    for i in range(Month):
        if i< 30:
            X['CPR'][i+1] = X['CPR'][i] + 0.002 * PSA / 100
        else:
            X['CPR'][i] = X['CPR'][i-1]
        
        X['SMM'][i] = 1 - (1-X['CPR'][i])**(1/12)
        if i ==0:
            X['FractionRemaining'][i] = (1-X['SMM'][i]) 
        else:
            X['FractionRemaining'][i] = X['FractionRemaining'][i-1] * (1-X['SMM'][i])
            
    return X

PSAB = PSA_function( Month , PSA )

# 1 Cashflow without prepayment
BeginningBlance = 337903338
CouponRate = 0.03404
ServiceFee = 0.00404

def CF( BeginningBlance , CouponRate , ServiceFee , Month , PSA):
    
    # Get PSA Performance from function PSA
    PSAB = PSA_function( Month , PSA)
    
    Beginningblance = np.zeros(Month)
    Payment = np.zeros(Month)
    Interest = np.zeros(Month)
    Servicing = np.zeros(Month)
    Principal = np.zeros(Month)
    EndingBlance = np.zeros(Month)
    SMM = np.zeros(Month)
    FractionRemaining = np.zeros(Month)
    TotalInterest = np.zeros(Month)
    TotalPrincipal = np.zeros(Month)
    Prepayment = np.zeros(Month)
    TotalPassthrough = np.zeros(Month)
    Blance = np.zeros(Month)
    
    # Beginning blance and Ending blance
    Beginningblance[0] = BeginningBlance
    EndingBlance[Month-1] = 0
    
    # The payment
    Payment[:] = BeginningBlance * CouponRate / 12 / (1 -(1+CouponRate/12)**(-Month))
    
    # SMM and FractionRemaining from PSAB
    SMM = PSAB['SMM']
    FractionRemaining = PSAB['FractionRemaining']
    
    # Get whole Beginning Blance and Ending Blance
    for i in range(1,Month):
        Beginningblance[i] = (Beginningblance[i-1]) * (1+CouponRate/12)- Payment[0]
        EndingBlance[i-1] = Beginningblance[i]
    
    # Get Interest payment every period
    # Payment subtract Interest part is the principal
    for i in range(Month):
        Interest[i] = Beginningblance[i] * (CouponRate-ServiceFee)/12
        Servicing[i] = Beginningblance[i] * ServiceFee/12
        Principal[i] = Payment[0] - Interest[i] - Servicing[i]
    
    TotalInterest[0] = Interest[0]
    TotalPrincipal[0] = Principal[0]
    for i in range(1,Month):
        TotalInterest[i] = FractionRemaining[i-1] * Interest[i]
        TotalPrincipal[i] = FractionRemaining[i-1] * Principal[i]
    
    Prepayment[0] = EndingBlance[0] * SMM[0]
    for i in range(1,Month):
        Prepayment[i] = EndingBlance[i] * FractionRemaining[i-1] * SMM[i]
    
    for i in range(0,Month):
        TotalPassthrough[i] = TotalInterest[i] + TotalPrincipal[i] + Prepayment[i]
        Blance[i] = Beginningblance[i] * FractionRemaining[i]
    
    #Construct X as a DataFrame to get the whole sheet
    X = pd.DataFrame(index = range(Month),columns=['BeginningBlance'])
    X['BeginningBlance'] = Beginningblance
    X['Payment'] = Payment
    X['Interest'] = Interest
    X['Servicing'] = Servicing
    X['Principal'] = Principal
    X['EndingBlance'] = EndingBlance
    X['SMM'] = SMM
    X['FractionRemaining'] = FractionRemaining
    X['TotalInterest'] = TotalInterest
    X['TotalPrincipal'] = TotalPrincipal
    X['Prepayment'] = Prepayment
    X['TotalPassthrough'] = TotalPassthrough
    X['Blance'] = Blance    
 
    return X

# We get the projection based on simple PSA model
Projection = CF( BeginningBlance , CouponRate , ServiceFee , Month , PSA)


# We are sure that we got the correct distribution
# Next, we need find the suitable PSA
# Get the historical prepayment
Prepayment = pd.read_csv('Prepayment.csv')

# Define function to calculate error
def Error( PSA , BeginningBlance , CouponRate , ServiceFee, Month , Prepayment):
    Projection = CF( BeginningBlance , CouponRate , ServiceFee , Month , PSA)
    ThreeMError = np.zeros(len(Prepayment))
    SixMError = np.zeros(len(Prepayment))
    
    for i in range(0,len(Prepayment)):
        if Prepayment['MA 3'][i] > 0:
            ThreeMError[i] = abs(Projection['Prepayment'][i] - Prepayment['MA 3'][i])
        if Prepayment['MA 6'][i] > 0:    
            SixMError[i] = abs(Projection['Prepayment'][i] - Prepayment['MA 6'][i])
    minError = min(sum(ThreeMError),sum(SixMError))
    return minError

# Get the optimize PSA based on our model
result = optimize.minimize( Error , PSA , args=( BeginningBlance , CouponRate , ServiceFee , Month , Prepayment ), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=None, tol=None, callback=None, options=None)
OPSA = result.x[0]  

# Then we get the projection based on best PSA
Projection = CF( BeginningBlance , CouponRate , ServiceFee , Month , OPSA)

# Get the plot of our outcome
Plot = pd.DataFrame(index = range(Month),columns=['Interest','Principal','Prepayment','Servicing'])
Plot['Interest'] = Projection['TotalInterest']
Plot['Principal'] = Projection['TotalPrincipal']
Plot['Prepayment'] = Projection['Prepayment']
Plot['Servicing'] = Projection['Servicing']
Plot.plot.area()

# 3 2 Interpolation
# We assume today is 2/15 because the spotrate curve we got is based on 2/15
# Got Libor

Libor = 1.444
KeyRate = STRIPS
length = np.max(KeyRate['Month to Maturity'])

keyratecurve = np.zeros(length+1)
keyratecurve[0] = Libor
for i in range(1,KeyRate['Month to Maturity'][0]+1):
    keyratecurve[i] = (KeyRate['Yield'][0] - Libor)/(KeyRate['Month to Maturity'][0]) * i + Libor

for i in range(1,len(KeyRate)):
    for j in range(KeyRate['Month to Maturity'][i-1]+1,KeyRate['Month to Maturity'][i]+1):
        keyratecurve[j] = ( KeyRate['Yield'][i] - KeyRate['Yield'][i-1])/(-KeyRate['Month to Maturity'][i-1]+KeyRate['Month to Maturity'][i]) * (j-KeyRate['Month to Maturity'][i-1]) + KeyRate['Yield'][i-1]

# Plot the interplation we got
plt.plot(KeyRate['Month to Maturity'], KeyRate['Yield'], 'ro')
plt.plot(keyratecurve,label = "Original")
plt.plot(0,Libor,'ro')

plt.plot(KeyRate['Month to Maturity'], KeyRate['Yield']+0.52, 'bo')
plt.plot(keyratecurve+0.52,label = "Shifted")
plt.plot(0,Libor+0.52,'bo')
plt.show()

# We need the projection from now on
# First projection at 3/25/2016, we assume that it is after 15 periods 
CleanProjection = Projection[14:]
Cashflow = np.array(CleanProjection['TotalPassthrough'])

# Discount the cashflow and get price

def GetPrice( Cashflow , keyratecurve ):
    DCF = np.zeros(len(Cashflow))
    Price = 0
    
    for i in range(len(Cashflow)):
        DCF[i] = Cashflow[i] * (1+keyratecurve[i+1]/1200)**(-i-1)
        Price = Price + DCF[i] 
    return Price

Price = GetPrice( Cashflow , keyratecurve + 0.52 )

# Calculate the DV01 and Modified Duration
def YTM( Cashflow , Price , freq = 12, guess=0.05):
    freq = float(freq)
    ytm_func = lambda y: \
        sum([ Cashflow[t] / (1+y/freq) ** (t+1) for t in range(0,len(Cashflow))] ) - Price
        
    return optimize.newton(ytm_func, guess)

ytm = YTM(Cashflow , Price , freq = 12, guess=0.05)

curve = np.zeros(length+1)
curve[:] = ytm *100
DV01 = - GetPrice( Cashflow , curve + 0.53 ) + GetPrice( Cashflow , curve + 0.52 )
Mod = (- GetPrice( Cashflow , curve + 0.53 ) + GetPrice( Cashflow , curve + 0.52 ))/ Price / 0.0001

# Q4
# Use avarage PSA changing levels to model
Change = pd.read_csv('PSAchange.csv') 

PSAs = np.zeros(9)
for i in range(len(PSAs)):
    PSAs[i] = Change.iloc[1][i+1] * OPSA + OPSA

DV01s = np.zeros(9) 
Mods = np.zeros(9)

for i in range (0,9):
    Projection = CF( BeginningBlance , CouponRate , ServiceFee , Month , PSAs[i])
    Cashflow = np.array(Projection[14:]['TotalPassthrough'])
    for j in range(len(Cashflow)):
        if Cashflow[j] > 0:
            Cashflow[j] = Cashflow[j]
        else:
            Cashflow[j] = 0
            
    Price = GetPrice( Cashflow , keyratecurve + 0.52 )
    ytm = YTM(Cashflow , Price , freq = 12, guess=0.05)
    
    curve = np.zeros(length+1)
    curve[:] = ytm *100
    DV01s[i] = - GetPrice( Cashflow , curve + 0.53 ) + GetPrice( Cashflow , curve + 0.52 )
    Mods[i] = (- GetPrice( Cashflow , curve + 0.53 ) + GetPrice( Cashflow , curve + 0.52 ))/ Price / 0.0001

