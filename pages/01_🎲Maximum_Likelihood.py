import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from app_common import *

st.set_page_config(
    page_title="Methane diffusive intrusion into a caprock",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="auto"
)

def isotopes_plot():
    D = D_ml
    t = t_ml
    delta = eval_delta(D, t, z, C12, C13, C12_res, C12_cap, C13_res, C13_cap)
    fig, ax1 = plt.subplots(figsize = (3,6))
    ax1.grid(which = 'minor')
    ax1.plot(delta, z, '-', label = 'Model evaluation best parameters')
    ax1.plot(delta_meas, z, '.', color = 'red', label = 'Measurements')
    ax1.set_xlabel(r'$\delta^{13}$C CH$_4$')
    ax1.set_xlim(-80, -20)
    ax1.set_ylabel('Distance from reservoir [m]')
    ax1.set_ylim(min(z), max(z))
    fig.legend()
    return fig

r"""

# 🎲 **Maximum Likelihood**
> Maximum likelihood framework tackles the estimation of model parameters. This procedure considers that the "true" values of model parameters are those
> that minimize the difference between measurements and model predictions (i.e., those parameters that minimize model residuals). In
> theory, residuals correspond to measurement errors. In practice, residuals are a combination of multiple sources of error (e.g., 
> measurement errors, analytical model errors, and numerical model errors). On the basis of the 
> central limit theorem, residuals are typically assumed to be normally distributed with zero mean. i.e.,

$$
p_{\varepsilon}(\varepsilon) = \frac{1}{\sqrt{2\pi\sigma_{\varepsilon}^2}}\exp{\left(-\frac{1}{2}\frac{\epsilon^2}{\sigma_{\varepsilon}^2}\right)}
$$

> As $\varepsilon = x^* - \hat{x}$ ($x^*$ being measured data and $\hat{x}$ being model predictions), then

$$
p_{\varepsilon}(x) = \frac{1}{\sqrt{2\pi\sigma_{\varepsilon}^2}}\exp{\left[-\frac{1}{2}\frac{(x^* - \hat{x})^2}{\sigma_{\varepsilon}^2}\right]}
$$

> The likelihood of the $n$ measured data is given by

$$
L(D,t|x^*) = \prod_{i=1}^{n}{\frac{1}{\sqrt{2\pi\sigma_{\varepsilon}^2}}\exp{\left[-\frac{1}{2}\frac{(x_i^* - \hat{x}_i)^2}{\sigma_{\varepsilon}^2}\right]}}
$$

> Which can be reduced to

$$
L(D,t|x^*) = \left(\frac{1}{\sqrt{2\pi\sigma_{\varepsilon}^2}}\right)^{n}\exp{\left[-\frac{1}{2\sigma_{\varepsilon}^2}\sum_{i=1}^{n}(x_i^* - \hat{x}_i)^2\right]}
$$

> The idea of maximum likelihood is to maximize the previous function (likelihood function, dah). Typically, 
> for the optimization procedure it is convenient to work with the logarithm instead of the actual values of the likelihood function. 
> Note that the maximum value of the logarithm of the function coincides with the maximum value of the function 
> (it makes sense). Also, since maximizing is for loosers (Weinberger, 2022), 
> we minimize the negative log-likelihood. i.e.,

$$
NLL(D,t|x^*) = \frac{n}{2}\ln(2\pi) + \frac{n}{2}\ln(\sigma_{\varepsilon}^2) + \frac{1}{2\sigma_{\varepsilon}^2}\sum_{i=1}^n{(x_i^*-\hat{x}_i)^2}
$$

> By considering that the variance of the error is independent of the measurements the function reduces to:

$$
\substack{\text{arg min}\\D\ t} \sum_{i=1}^n{(x_i^*-\hat{x}_i)^2}
$$
> which is known as loss function. The optimization procedure can be performed by either analytical or numerical methods. 
> If the function is not crazy it is convenient to use the analytical procedure. Such a procedure consists of deriving the loss function 
> with respect to the parameter to be estimated and making the derivative equal to zero to estimate the parameter value that minimizes the loss 
> function. The advantage of the analytical 
> method is that it provides a closed-form solution of the optimal parameter values whose evaluation has a virtually zero computational cost. 
> If the analytical equations are intractable, it is often convenient to find the best parameters by randomly sampling in the parameter 
> space of the uncertain parameters. Such a procedure has two drawbacks: (i) There is no guarantee that the optimal parameter values 
> can be found; 
> and (ii) the procedure is typically associated with a high computational cost since the model has to be evaluated several times.

> In the following you can define the number of parameter sets to be evaluated in order to find the parameter values associated with 
> the minimum of the loss function.
"""

data_file = st.file_uploader("Measurements_file", "txt", False, key="all_data")
measurements = np.loadtxt(data_file, skiprows=1)
z = measurements[:,0]
delta_meas = measurements[:,1]

C12 = np.ones(z.size)
C13 = np.ones(z.size)

delta_res = st.sidebar.number_input('delta_13_methane reservoir', -60.0, -40.0, -51.0)
delta_cap = st.sidebar.number_input('delta_13_methane caprock', -75.0, -50.0, -70.0)
C_res = st.sidebar.number_input('Methane concentration reservoir (ppm)', 100.0, 100000.0, 2000.0)
C_cap = st.sidebar.number_input('Methane concentration caprock (ppm)', 100.0, 100000.0, 1000.0)

Sim = st.number_input('Number of simulations', 10, 1000000, 1000)

R_res = (1+delta_res/1000)*0.011237
R_cap = (1+delta_cap/1000)*0.011237

C12_res = C_res/(1+R_res)
C12_cap = C_cap/(1+R_cap)

C13_res = C_res*R_res/(1+R_res)
C13_cap = C_cap*R_cap/(1+R_cap)

col1,col2,col3 = st.columns([2,2,2])

if st.button('Run Maximum Likelihood'):
    D_ml, t_ml = execute_ML(Sim, delta_meas, z, C12, C13, C12_res, C12_cap, C13_res, C13_cap)
    ss = np.var((eval_delta(D_ml,t_ml, z, C12, C13, C12_res, C12_cap, C13_res, C13_cap))-delta_meas)
    st.write('The variance of the residuals is: ', ss)
    st.write('The ML estimate of D is: ', D_ml, ' m^2/s')
    st.write('The ML estimate of t is: ', t_ml, ' millions of years')
    with col2:
        st.pyplot(isotopes_plot())
