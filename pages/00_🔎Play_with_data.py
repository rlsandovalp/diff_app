import numpy as np
import matplotlib.pyplot as plt
from app_common import *
import streamlit as st

st.set_page_config(
    page_title="Play with data",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="auto"
)

def isotopes_plot(controlContainer):
    with controlContainer:
        st.markdown("<br>"*3,unsafe_allow_html=True)
        with st.expander("Fitting parameters:",expanded=True):
            D = st.number_input('D', 1E-12, 1E-8, 1E-9, format = '%e')
            t = st.number_input("t", 0.1, 50.0, 5.0)
    fig, ax1 = plt.subplots()
    ax1.grid(which = 'minor')
    delta = eval_delta(D, t, z, C12, C13, C12_res, C12_cap, C13_res, C13_cap)
    ax1.plot(delta, z, '-', label = 'Model evaluation')
    ax1.plot(delta_meas, z, '.', color = 'red', label = 'Measurements')
    ax1.set_xlabel(r'$\delta^{13}$C CH$_4$')
    ax1.set_xlim(-80, -20)
    ax1.set_ylabel('Distance from reservoir [m]')
    ax1.set_ylim(min(z), max(z))
    ax1.legend()
    return fig

r"""

# 🔎 **Play with the parameter values**
> We will analyze the diffusion intrusion of methane into a caprock. This process is described by the diffusion equation.

$$
    \frac{\partial C}{\partial t} = D\nabla C
$$

> Considering a semi-infinite domain, in which the methane concentration at the beggining of the simulation is constant across 
> the entire caprock (i.e., $C(z,0) = C_{cap}$), and considering that the methane concentration in the reservoir is constant 
> during the entire simulation (i.e., $C(0,t) = C_{res}$), the following analytical solution can be employed to estimate the
> methane concentration $C$ at a given position $z$ and a given time $t$.

$$
    C(z,t) = \left(1-\text{erf}\left[\frac{z}{2\sqrt{Dt}}\right]\right)(C_{res}-C_{cap})+C_{cap}
$$
> where $D$ is the effective difussion coefficient, and $\text{erf}[\ ]$ is the error function.

Here you can adjust the value of the two model parameters (i.e., $D$ and $t$) and compare the analytical solution computed 
with those parameter values with the measurements of $\delta^{13}$CH$_4$ collected in field.
"""

delta_res = st.sidebar.number_input('delta_13_methane reservoir', -60.0, -40.0, -51.0)
delta_cap = st.sidebar.number_input('delta_13_methane caprock', -75.0, -50.0, -70.0)
C_res = st.sidebar.number_input('Methane concentration reservoir (ppm)', 100.0, 100000.0, 2000.0)
C_cap = st.sidebar.number_input('Methane concentration caprock (ppm)', 100.0, 100000.0, 1000.0)

R_res = (1+delta_res/1000)*0.011237
R_cap = (1+delta_cap/1000)*0.011237

C12_res = C_res/(1+R_res)
C12_cap = C_cap/(1+R_cap)

C13_res = C_res*R_res/(1+R_res)
C13_cap = C_cap*R_cap/(1+R_cap)

data_file = st.file_uploader("Measurements_file", "txt", False, key="all_data")

# if st.button('Plot data'):
measurements = np.loadtxt(data_file, skiprows=1)
z = measurements[:,0]
delta_meas = measurements[:,1]

C12 = np.ones(z.size)
C13 = np.ones(z.size)
col1,col2 = st.columns([2,2])

with col2:
    st.pyplot(isotopes_plot(col1))