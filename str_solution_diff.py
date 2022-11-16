import streamlit as st

st.set_page_config(
    page_title="Diffusion in Caprocks",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

r"""
# Methane Diffusion in Caprocks
"""

"""
## Objectives 
> The objective of this app is to ilustrate the calibration procedure 
> employing a deterministic approach (maximum likelihood) and a 
> stochastic approach (acceptance rejection algorithm).

> The calibration procedure is explained with the example of diffusive 
> flow in a caprock (i.e., a geomaterial of low permeability). In the page 
> 00 you will be able to adjust the parameter values of the diffusive model 
> and see how the prediction of the model varies. In the page 01, the maximum 
> likelihood procedure is explained and you will be able to perform the 
> maximum likelihood calibration of the model parameters. In the page 02, the 
> acceptance-rejection algorithm is explained and model parameters calibration 
> is performed. 


## Ideas to implement
- Perform a Sensitivity Analysis of the Diffusion Equation.
- Check if it is possible to perform an analysis with "molecule fractionation".
- Perform an stochastic callibration of model parameters with the available data.


"""