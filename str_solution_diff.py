import numpy as np
import math
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Methane diffusive intrusion into a caprock",
    page_icon="👷‍♂️",
    layout="wide",
    initial_sidebar_state="auto"
)

def eval_delta(DC12,t, z):
    time = t*1000000*365*86400
    DC13 = DC12*0.996

    for z_pos, z_value in enumerate(z):
        C12[z_pos] = (1-math.erf(z_value/(2*(DC12*time)**0.5)))*(C12_res-C12_cap)+C12_cap
        C13[z_pos] = (1-math.erf(z_value/(2*(DC13*time)**0.5)))*(C13_res-C13_cap)+C13_cap
    
    return ((C13/C12)/0.0112372-1)*1000

def isotopes_plot(controlContainer):
    with controlContainer:
        st.markdown("<br>"*3,unsafe_allow_html=True)
        with st.expander("Fitting parameters:",expanded=True):
            D = st.number_input('D', 1E-12, 1E-8, 1E-9, format = '%e')
            t = st.slider("t",0.1,30.0,5.0)

    fig, ax1 = plt.subplots()
    ax1.grid(which = 'minor')
    delta = eval_delta(D, t, z)
    ax1.plot(delta, z, '-', label = 'Model evaluation')
    ax1.plot(delta_meas, z, '.', color = 'red', label = 'Measurements')
    ax1.set_xlabel(r'$\delta^{13}$C CH$_4$')
    ax1.set_xlim(-80, -20)
    ax1.set_ylabel('Distance from reservoir [m]')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Depth [m]')
    ax2.set_ylim(1424, 746)
    ax1.set_ylim(0, 678)
    ax1.legend()
    return fig

def histogram_D():
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Diffusion Coefficient [M2/S]')
    ax1.set_ylabel('Frequency')
    plt.hist(x=D_pos, bins='auto')
    return fig

def histogram_t():
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Intrusion time [MY]')
    ax1.set_ylabel('Frequency')
    plt.hist(x=t_pos, bins='auto')
    return fig

def stochastic_solutions_plot():
    fig, ax1 = plt.subplots(figsize = (3,6))
    ax1.grid(which = 'minor')
    for i in range(len(D_pos)):
        ax1.plot(stochastic_solutions[:,i], z, '-', color = 'gray')
    ax1.plot(stochastic_solutions.mean(axis = 1), z, '-')
    ax1.plot(delta_meas, z, '.', color = 'red')
    ax1.set_xlabel(r'$\delta^{13}$C CH$_4$')
    ax1.set_xlim(-80, -20)
    ax1.set_ylabel('Distance from reservoir [m]')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Depth [m]')
    ax2.set_ylim(1424, 746)
    ax1.set_ylim(0, 678)
    return fig   

r"""

# 👷‍♂️ **Methane diffusive intrusion into a caprock**
We will analyze the diffusion intrusion of methane into a caprock. This process is described by de diffusion equation.

$$
    \frac{\partial C}{\partial t} = D\nabla C
$$

In this first stage we will use the solution for a semi-infinite domain.

$$
    C(z,t) = \left(1-\text{erf}\left[\frac{z}{2\sqrt{Dt}}\right]\right)(C_{res}-C_{cap})+C_{cap}
$$

Where $C_{cap}$ is the methane concentration in the caprock at time $t=0$. 
$C_{res}$ is the methane concentration in the reservoir,this concentration is considered to be constant during the entire period of analysis. 
$D$ is the effective difussion coefficient. 
$\text{erf}[\ ]$ is the error function.

"""

measurements = np.loadtxt('assets/data.txt', skiprows=1, delimiter = ',')
z = measurements[:,0]
delta_meas = measurements[:,1]

C12 = np.ones(z.size)
C13 = np.ones(z.size)

delta_res = st.sidebar.number_input('delta_13_methane reservoir', -60.0, -40.0, -45.0)
delta_cap = st.sidebar.number_input('delta_13_methane caprock', -70.0, -50.0, -60.0)
C_res = st.sidebar.number_input('Methane concentration reservoir (ppm)', 100.0, 100000.0, 10000.0)
C_cap = st.sidebar.number_input('Methane concentration caprock (ppm)', 100.0, 100000.0, 1000.0)


R_res = (1+delta_res/1000)*0.011237
R_cap = (1+delta_cap/1000)*0.011237

C12_res = C_res/(1+R_res)
C12_cap = C_cap/(1+R_cap)

C13_res = C_res*R_res/(1+R_res)
C13_cap = C_cap*R_cap/(1+R_cap)

col1,col2 = st.columns([1,2.5])

with col2:
    st.pyplot(isotopes_plot(col1))


r"""
****
### Stochastic calibration employing Aceptance-rejection sampling
Since the data collected in laboratory is always associated with an error of acquisition, the interpretative 
models are not perfect, natural media is heterogeneous, bla bla bla. It is convenient characterize the model uncertain
 parameters recognizing this challenging conditions, therefore ... Acceptance rejection

First, we need to define some parameters associated with the algorithm
"""

Sim = st.number_input('Number of simulations A/R sampling', 100, 100000, 10000)
s2_obs = st.number_input("Standard error of the observations", 0.1, 10.0, 10.0)


D_pos = []
t_pos = []


for i in range(Sim):
    DC12 = np.random.uniform(low=1E-12, high=1E-9)
    t = np.random.uniform(low=0.1, high=8)

    time = t*1000000*365*86400
    DC13 = DC12*0.996

    for z_pos, z_value in enumerate(z):
        C12[z_pos] = (1-math.erf(z_value/(2*(DC12*time)**0.5)))*(C12_res-C12_cap)+C12_cap
        C13[z_pos] = (1-math.erf(z_value/(2*(DC13*time)**0.5)))*(C13_res-C13_cap)+C13_cap

    delta = ((C13/C12)/0.0112372-1)*1000

    alpha = math.exp(-(delta_meas-delta)@(delta_meas-delta)/(2*s2_obs))
    u = np.random.uniform()
    if alpha > u:
        D_pos.append(DC12)
        t_pos.append(t)


stochastic_solutions = np.zeros((z.size,len(D_pos)))

for i in range(len(D_pos)):
    DC12 = D_pos[i]
    t = t_pos[i]

    time = t*1000000*365*86400
    DC13 = DC12*0.996

    for z_pos, z_value in enumerate(z):
        C12[z_pos] = (1-math.erf(z_value/(2*(DC12*time)**0.5)))*(C12_res-C12_cap)+C12_cap
        C13[z_pos] = (1-math.erf(z_value/(2*(DC13*time)**0.5)))*(C13_res-C13_cap)+C13_cap
    
    stochastic_solutions[:,i] = ((C13/C12)/0.0112372-1)*1000

col3,col4,col5 = st.columns([3,2,2])

with col3:
    st.pyplot(stochastic_solutions_plot())

with col4:
    st.pyplot(histogram_t())

with col5:
    st.pyplot(histogram_D())
