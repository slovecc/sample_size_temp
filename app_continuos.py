### missing : one tail vs two tails
### implement the formula scratch rather than python
### include the DAU and length of experiment if included

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import norm
import streamlit as st
import seaborn as sns
from statsmodels.stats.power import TTestIndPower
from aux_funct import *

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title="AB test sample size calculator"
    )

################################################
##### sidebar section
################################################
# Sidebar - continous vs fractional
st.sidebar.markdown(
    """
### Nature of the metric
Choose if you need a continous or fractional metric
"""
)
metric_type = st.sidebar.radio("Metric type", ("Continous", "Fractional"), 
                                index=1)


# Sidebar - alpha
st.sidebar.markdown(
    """
### Significance level
95% is often used as the threshold before a result is declared as
statistically significant.
"""
)


alpha = 1 - st.sidebar.slider(
    "Significance level",
    value=0.95,
    min_value=0.5,
    max_value=0.99,
)

# Sidebar - beta
st.sidebar.markdown(
    """
### Statistical power
80% is generally accepted as the minimum required power level.
"""
)
beta = 1 - st.sidebar.slider("Power", value=0.8, min_value=0.5, max_value=0.99)

st.sidebar.markdown(
    """
    ### One vs. two tails
"""
)

# Sidebar - number of tails
num_tails = st.sidebar.radio("Number of tails", ("One", "Two"), index=1)

# Sidebar - number of variant
numb_variant = st.sidebar.number_input("Number of variant", min_value=2, max_value=10, value=2, step=1)



##### main section

"""
# THIS IS A TEMP DRAFT FOR AB Test Sample Size 
Input the expected daily observations and conversions to return a plot
containing potential runtimes and their associated minimum detectable effect.
"""

"""
## Baselines:
"""

if metric_type == 'Continous' :
    
    f"Average historical of the metric"
    kpi_mean = st.number_input("Average", value=0.024, step=0.01)
    
    
    f"Average standard deviation of the metric"
    kpi_std = st.number_input("Standard deviation", value=0.165, step=0.01)
    
    # OPTIONAL
    mde = st.slider("Minimum Detectable Effect (%)", value=0.1, min_value=0.01, max_value=50.)

    """
    ## Results:
    """
    
    f"Main result for the sample size"
    diff_mu = (1+mde)*kpi_mean - kpi_mean
    size = diff_mu/kpi_std
    n=TTestIndPower().solve_power(effect_size = size, 
                                     power = 1-beta, 
                                     alpha = alpha)
    
    st.success("The sample size for the kpi with mean: {} std dev: {} and significance: {:.2%}"
               " power : {:.2%} at {}% of increase is equals to {}".format(kpi_mean,kpi_std,alpha,
                                                                           (1-beta), mde ,round(n)))

    
    f"Table with different scenarios"
    scenario = []

    for increase in [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5] :
        diff_mu = (1+increase)*kpi_mean - kpi_mean
        size = diff_mu/kpi_std
        n=TTestIndPower().solve_power(effect_size = size, 
                                         power = beta, 
                                         alpha = alpha)
        
        scenario.append((increase, size, beta, alpha, n))
        
    df_scenario=pd.DataFrame(scenario, columns=('increase[%]','size', 'power', 'significance','sample'))
    df_scenario[['increase[%]']] = df_scenario[['increase[%]']] *100

    st.write(df_scenario)



    if st.checkbox("Show sentitivity analysis"):
         """
         ### include the sensitivity analysis:
         """
         perc_increase = np.arange(0.001,0.05,0.001)
         #effect_size=diff_mu/sigma
         power = [0.7,0.8,0.9]  
         alpha = [0.01,0.02,0.05]  
         dn = []
         
         for increase in perc_increase :
             for p in power : 
                 for sig in alpha:
                     diff_mu = (1+increase)*kpi_mean - kpi_mean
                     size = diff_mu/kpi_std
                     n=TTestIndPower().solve_power(effect_size = size, 
                                                      power = p, 
                                                      alpha = sig)
                     
                     dn.append((increase,size, p, sig,n))
         
         pn=pd.DataFrame(dn, columns=('increase','size', 'power', 'significance','sample'))
         pn.power=pn.power.astype(np.float16)
         powerr = np.array([0.7,0.8,0.9])
         signn = np.array([0.01,0.02,0.05])
         

         #sns.set(font_scale = 2)
         

         fig = plt.xlabel("lift ")
         
         sns.lineplot(data=pn[(pn.power == 0.7) & (pn.significance == 0.05)], 
                      x="increase", y="sample",color='blue', linewidth = 2.5, marker='o',label='b = 0.7,a=0.05')
         
         sns.lineplot(data=pn[(pn.power == 0.8) & (pn.significance == 0.05)], 
                      x="increase", y="sample",color='red', linewidth = 2.5, marker='o',label='b = 0.8,a=0.05')
         
         sns.lineplot(data=pn[(pn.power == 0.85) & (pn.significance == 0.05)], 
                     x="increase", y="sample",color='green', linewidth = 2.5, marker='o',label='b = 0.85,a=0.05')
         
         
         sns.lineplot(data=pn[(pn.power == 0.7) & (pn.significance == 0.01)], 
                      x="increase", y="sample",color='blue', linewidth = 2.5, marker='s',label='b = 0.7,a=0.01')
         
         sns.lineplot(data=pn[(pn.power == 0.8) & (pn.significance == 0.01)], 
                      x="increase", y="sample",color='red', linewidth = 2.5, marker='s',label='b = 0.8,a=0.01')
         
         sns.lineplot(data=pn[(pn.power == 0.85) & (pn.significance == 0.01)], 
                      x="increase", y="sample",color='green', linewidth = 2.5, marker='s',label='b = 0.85,a=0.01')
         
         fig = plt.legend(fontsize='12')
         fig = plt.yscale("log")
         
         min_y = pn[['sample']].min().item()
         max_y = pn[['sample']].max().item()
         
         fig = plt.ylim(min_y, max_y)

         
         st.pyplot()
         
         
    if st.checkbox("Do you want to know how long will stands the experiment (in weeks)? Include the DAU"):
         """
         ### TO BE DONE: include the week length
         """

                    
            
            