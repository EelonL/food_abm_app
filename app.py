import streamlit as st
import pandas as pd

from food_abm.model import FoodSupplyModel
from food_abm.scenarios import SCENARIOS

st.title("Food ABM App")

scenario_name = st.selectbox("Valitse skenaario", list(SCENARIOS.keys()))
steps = st.slider("Aikajaksojen määrä", 10, 100, 30)
num_agents = st.slider("Pk-agenttien määrä", 10, 200, 50)

if st.button("Aja simulaatio"):
    params = SCENARIOS[scenario_name].copy()
    model = FoodSupplyModel(num_smes=num_agents, steps=steps, **params)

    for _ in range(steps):
        model.step()

    results = model.datacollector.get_model_vars_dataframe()
    st.dataframe(results.tail())
    st.line_chart(results)
