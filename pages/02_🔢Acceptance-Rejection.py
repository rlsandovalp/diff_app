import numpy as np
import math
import matplotlib.pyplot as plt
import streamlit as st
from app_common import *

st.set_page_config(
    page_title="Acceptance-Rejection",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="auto"
)

def execute_sc():
    D_pos = []
    t_pos = []
    D_sampled = []
    t_sampled = []
    D_ml, t_ml = execute_ML(Sim, delta_meas, z, C12, C13, C12_res, C12_cap, C13_res, C13_cap)
    delta_ml = eval_delta(D_ml,t_ml, z, C12, C13, C12_res, C12_cap, C13_res, C13_cap)
    Max_like = (1/math.sqrt(2*math.pi*ss))**len(delta_meas)*math.exp(-((delta_meas-delta_ml)@(delta_meas-delta_ml))/(2*ss))
    for i in range(Sim):
        DC12 = np.random.uniform(low=1E-11, high=1E-10)
        t = np.random.uniform(low=0.1, high=8)
        delta = eval_delta(DC12, t, z, C12, C13, C12_res, C12_cap, C13_res, C13_cap)
        alpha = (1/math.sqrt(2*math.pi*ss))**len(delta_meas)*math.exp(-((delta_meas-delta)@(delta_meas-delta))/(2*ss))/Max_like
        u = np.random.uniform()
        D_sampled.append(DC12)
        t_sampled.append(t)
        if alpha > u:
            D_pos.append(DC12)
            t_pos.append(t)

    stochastic_solutions = np.zeros((z.size,len(D_pos)))

    for i in range(len(D_pos)):
        DC12 = D_pos[i]
        t = t_pos[i]
        stochastic_solutions[:,i] = eval_delta(DC12, t, z, C12, C13, C12_res, C12_cap, C13_res, C13_cap)
    return D_pos, t_pos, D_sampled, t_sampled, stochastic_solutions

def histogram_D():
    fig, ax = plt.subplots(2,1,figsize = (3,6))
    ax[0].set_ylabel('Frequency')
    ax[0].hist(x=D_sampled, bins='auto')
    ax[1].set_xlabel('Diffusion Coefficient [M2/S]')
    ax[1].set_ylabel('Frequency')
    ax[1].hist(x=D_pos, bins='auto')
    return fig

def histogram_t():
    fig, ax = plt.subplots(2,1,figsize = (3,6))
    ax[0].set_ylabel('Frequency')
    ax[0].hist(x=t_sampled, bins='auto')
    ax[1].set_xlabel('Intrusion time [MY]')
    ax[1].set_ylabel('Frequency')
    ax[1].hist(x=t_pos, bins='auto')
    return fig

def stochastic_solutions_plot():
    fig, ax1 = plt.subplots(figsize = (3,6))
    ax1.grid(which = 'minor')
    for i in range(len(D_pos)):
        ax1.plot(stochastic_solutions[:,i], z, '-', color = 'gray')
    ax1.plot(stochastic_solutions.mean(axis = 1), z, '-')
    ax1.plot(delta_meas, z, '.', color = 'red')
    ax1.set_xlabel(r'$\delta^{13}$C CH$_4$')
    ax1.text(0.2, 0.9, 'Acceptance rate = '+str(len(D_pos)*100/Sim)+'%', transform=ax1.transAxes)
    ax1.text(0.2, 0.8, 'Accepted samples = '+str(len(D_pos)), transform=ax1.transAxes)
    ax1.set_xlim(-80, -20)
    ax1.set_ylabel('Distance from reservoir [m]')
    ax1.set_ylim(min(z), max(z))
    return fig   


r"""
****
# 🔢 **Stochastic calibration employing Aceptance-rejection sampling**
> Since the data collected in laboratory is always associated with an error of acquisition, the interpretative 
> models are not perfect, natural media is heterogeneous, and many other resons; it is convenient to characterize the model uncertain
> parameters under a bayesian framework (i.e., recognize that the model parameters are not deterministic but probabilistic 
> quantities). The most simple (and most inneficient) algorithm is called Acceptance-Rejection sampling. The idea of the algorithm is 
> to sample from the prior pdf of the model uncertain parameters and accept the parameter combination provided that the likelihood 
> function meets some conditions.

Remember that the likelihood function can be formulated as:

$$
L(D,t|x^*) = \left(\frac{1}{\sqrt{2\pi\sigma_{\varepsilon}^2}}\right)^{n}\exp{\left[-\frac{1}{2\sigma_{\varepsilon}^2}\sum_{i=1}^{n}(x_i^* - \hat{x}_i)^2\right]}
$$

where $\hat{x}$ is the value of the model evaluated for a given $D$ and $t$. Thus, the maximum likelihood value is

$$
L(D_{ML},t_{ML}|x^*)
$$

The Acceptance-Rejection sampling algorithm consist in repeat several times the following procedure:

1) Draw random values ($D_i$ and $t_i$) from the ranges of variability of the model uncertain parameters
2) Evaluate the model and compute the normalized likelihood function, $\alpha$

$$
\alpha = \frac{L(D_i,t_i|x^*)}{L(D_{ML},t_{ML}|x^*)}
$$

That can be reformulated as

$$
\alpha = \frac{\exp{\left[-\frac{1}{2\sigma_{\varepsilon}^2}\sum_{i=1}^{n}(x_i^* - x(D,t))^2\right]}}{\exp{\left[-\frac{1}{2\sigma_{\varepsilon}^2}\sum_{i=1}^{n}(x_i^* - x(D_{ML},t_{ML}))^2\right]}}
$$

Note that $\alpha$ varies in the interval [0,1]

3) Draw a random value $u_i$ from the uniform distribution in the interval [0,1].
4) If $\alpha_i$ is larger than $u_i$, then accept $D_i$ and $t_i$. Otherwise reject the parameter set and start again.


In the following you are asked to define the number of iterations of the Acceptance-Rejection algorithm and the variance of the 
observation error.

"""


data_file = st.file_uploader("Measurements_file", "txt", False, key="all_data")
measurements = np.loadtxt(data_file, skiprows=1)
z = measurements[:,0]
delta_meas = measurements[:,1]

C12 = np.ones(z.size)
C13 = np.ones(z.size)

delta_res = st.sidebar.number_input('delta_13_methane reservoir', -60.0, -40.0, -51.0)
delta_cap = st.sidebar.number_input('delta_13_methane caprock', -70.0, -50.0, -70.0)
C_res = st.sidebar.number_input('Methane concentration reservoir (ppm)', 100.0, 100000.0, 2000.0)
C_cap = st.sidebar.number_input('Methane concentration caprock (ppm)', 100.0, 100000.0, 1000.0)

R_res = (1+delta_res/1000)*0.011237
R_cap = (1+delta_cap/1000)*0.011237

C12_res = C_res/(1+R_res)
C12_cap = C_cap/(1+R_cap)

C13_res = C_res*R_res/(1+R_res)
C13_cap = C_cap*R_cap/(1+R_cap)

Sim = st.number_input('Number of simulations A/R sampling', 100, 100000, 100)
ss = st.number_input("Standard error of the observations", 0.01, 50.0, 1.0)

if st.button('Run Acceptance-Rejection'):
    D_pos, t_pos, D_sampled, t_sampled, stochastic_solutions = execute_sc()
    col3,col4,col5 = st.columns([2,2,2])

    with col3:
        st.pyplot(stochastic_solutions_plot())

    with col4:
        st.pyplot(histogram_t())

    with col5:
        st.pyplot(histogram_D())


