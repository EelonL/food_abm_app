from __future__ import annotations

import copy
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from food_abm.model import FoodSupplyModel
from food_abm.scenarios import SCENARIOS
from food_abm.archetypes import SME_ARCHETYPES


st.set_page_config(
    page_title="Food ABM App",
    page_icon="🌾",
    layout="wide",
)

st.title("Food ABM App")
st.caption(
    "Agenttipohjainen simulaatiomalli Uudenmaan ruokaketjun pk-yritysten "
    "logistiikan, kanavien ja skenaarioiden tarkasteluun."
)


# ---------- Helpers ----------

KEY_METRICS = [
    "AvgProfit",
    "AvgLogisticsCost",
    "AvgDependency",
    "SurvivalRate",
    "AvgChannelCount",
    "TotalEmissions",
]


def clone_scenario(name: str):
    """Return a shallow copy of a scenario dataclass so UI edits don't mutate globals."""
    return copy.copy(SCENARIOS[name])


def run_model_once(scenario_name: str, n_smes: int, steps: int, seed: int):
    scenario = clone_scenario(scenario_name)
    model = FoodSupplyModel(n_smes=n_smes, scenario=scenario, seed=seed)
    model.run_model(steps=steps)
    model_df = model.get_model_dataframe().copy()
    agent_df = model.get_agent_dataframe().copy()

    if not model_df.empty:
        model_df = model_df.reset_index().rename(columns={"index": "Step"})
    if not agent_df.empty:
        # Mesa agent dataframe often has MultiIndex [Step, AgentID]
        if isinstance(agent_df.index, pd.MultiIndex):
            agent_df = agent_df.reset_index()
        else:
            agent_df = agent_df.reset_index().rename(columns={"index": "AgentID"})

    return model_df, agent_df


def run_scenario_grid(
    selected_scenarios: List[str], n_smes: int, steps: int, seeds: List[int]
) -> pd.DataFrame:
    rows = []
    for scenario_name in selected_scenarios:
        for seed in seeds:
            model_df, _ = run_model_once(
                scenario_name=scenario_name,
                n_smes=n_smes,
                steps=steps,
                seed=seed,
            )
            if model_df.empty:
                continue
            final_row = model_df.iloc[-1].copy()
            final_row["Scenario"] = scenario_name
            final_row["Seed"] = seed
            rows.append(final_row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def plot_time_series(model_df: pd.DataFrame, metric: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(model_df["Step"], model_df[metric], marker="o")
    ax.set_title(f"{metric} ajan yli")
    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    return fig


def plot_metric_bars(results_df: pd.DataFrame, metric: str):
    grouped = results_df.groupby("Scenario", as_index=False)[metric].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(grouped["Scenario"], grouped[metric])
    ax.set_title(f"{metric} skenaarioittain")
    ax.set_xlabel("Skenaario")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def plot_tradeoff(results_df: pd.DataFrame):
    grouped = results_df.groupby("Scenario", as_index=False).agg(
        AvgProfit=("AvgProfit", "mean"),
        TotalEmissions=("TotalEmissions", "mean"),
        SurvivalRate=("SurvivalRate", "mean"),
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = 300 * grouped["SurvivalRate"].clip(lower=0.05)
    ax.scatter(grouped["AvgProfit"], grouped["TotalEmissions"], s=sizes)
    for _, row in grouped.iterrows():
        ax.annotate(
            row["Scenario"],
            (row["AvgProfit"], row["TotalEmissions"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_title("Trade-off: kate vs päästöt")
    ax.set_xlabel("AvgProfit")
    ax.set_ylabel("TotalEmissions")
    ax.grid(True, alpha=0.3)
    return fig


def plot_archetype_boxplot(agent_df: pd.DataFrame, metric: str = "LastProfit"):
    plot_df = agent_df.copy()
    if "Archetype" not in plot_df.columns or metric not in plot_df.columns:
        return None

    grouped = [
        plot_df.loc[plot_df["Archetype"] == arch, metric].dropna().values
        for arch in plot_df["Archetype"].dropna().unique()
    ]
    labels = list(plot_df["Archetype"].dropna().unique())

    if not grouped:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(grouped, labels=labels)
    ax.set_title(f"{metric} arkkityypeittäin")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def metric_card(label: str, value, help_text: str = ""):
    st.metric(label=label, value=value, help=help_text)


# ---------- Sidebar ----------

st.sidebar.header("Asetukset")

mode = st.sidebar.radio(
    "Tila",
    ["Yksittäinen ajo", "Skenaariovertailu"],
    index=0,
)

n_smes = st.sidebar.slider("Pk-agenttien määrä", min_value=12, max_value=180, value=48, step=6)
steps = st.sidebar.slider("Stepien määrä", min_value=6, max_value=60, value=18, step=2)
seed = st.sidebar.number_input("Satunnaissiementä (single run)", min_value=1, max_value=100000, value=42, step=1)

with st.sidebar.expander("Näytä pk-arkkityypit", expanded=False):
    for name, params in SME_ARCHETYPES.items():
        st.write(f"**{name}**")
        st.json(params)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Vinkki: aloita baseline-, neutral_hub- ja multichannel-skenaarioilla."
)


# ---------- Main UI ----------

if mode == "Yksittäinen ajo":
    st.subheader("Yksittäinen skenaarioajo")

    scenario_name = st.selectbox(
        "Valitse skenaario",
        options=list(SCENARIOS.keys()),
        index=0,
    )

    if st.button("Aja simulaatio", type="primary"):
        with st.spinner("Simulaatio käynnissä..."):
            model_df, agent_df = run_model_once(
                scenario_name=scenario_name,
                n_smes=n_smes,
                steps=steps,
                seed=int(seed),
            )

        if model_df.empty:
            st.error("Mallista ei tullut tuloksia.")
        else:
            final_row = model_df.iloc[-1]

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Keskimääräinen kate", f"{final_row['AvgProfit']:.2f}")
                metric_card("Keskimääräinen riippuvuus", f"{final_row['AvgDependency']:.2f}")
            with c2:
                metric_card("Logistiikkakustannus", f"{final_row['AvgLogisticsCost']:.2f}")
                metric_card("Kanavien määrä", f"{final_row['AvgChannelCount']:.2f}")
            with c3:
                metric_card("Säilymisaste", f"{100*final_row['SurvivalRate']:.1f}%")
                metric_card("Kokonaispäästöt", f"{final_row['TotalEmissions']:.2f}")

            tab1, tab2, tab3 = st.tabs(["Aikasarjat", "Arkkityypit", "Data"])

            with tab1:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.pyplot(plot_time_series(model_df, "AvgProfit"))
                    st.pyplot(plot_time_series(model_df, "SurvivalRate"))
                with col_b:
                    st.pyplot(plot_time_series(model_df, "AvgDependency"))
                    st.pyplot(plot_time_series(model_df, "TotalEmissions"))

            with tab2:
                fig = plot_archetype_boxplot(agent_df, metric="LastProfit")
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.info("Arkkityyppikohtaista boxplot-kuvaa ei voitu muodostaa.")

                fig2 = plot_archetype_boxplot(agent_df, metric="LastDependencyRatio")
                if fig2 is not None:
                    st.pyplot(fig2)

            with tab3:
                st.write("**Mallitaso (viimeiset rivit)**")
                st.dataframe(model_df.tail(), use_container_width=True)
                st.write("**Agenttitaso (viimeiset rivit)**")
                st.dataframe(agent_df.tail(20), use_container_width=True)

else:
    st.subheader("Skenaariovertailu")

    selected_scenarios = st.multiselect(
        "Valitse vertailtavat skenaariot",
        options=list(SCENARIOS.keys()),
        default=["baseline", "neutral_hub", "multichannel"],
    )
    iterations = st.slider("Toistojen määrä per skenaario", min_value=2, max_value=20, value=6, step=1)
    base_seed = st.number_input("Aloitussiemen", min_value=1, max_value=100000, value=100, step=1)

    if st.button("Aja skenaariovertailu", type="primary"):
        if not selected_scenarios:
            st.warning("Valitse vähintään yksi skenaario.")
        else:
            seeds = list(range(int(base_seed), int(base_seed) + iterations))

            with st.spinner("Vertailu käynnissä..."):
                results_df = run_scenario_grid(
                    selected_scenarios=selected_scenarios,
                    n_smes=n_smes,
                    steps=steps,
                    seeds=seeds,
                )

            if results_df.empty:
                st.error("Vertailusta ei tullut tuloksia.")
            else:
                grouped = results_df.groupby("Scenario", as_index=False)[KEY_METRICS].mean()

                st.write("### Skenaarioiden keskiarvot")
                st.dataframe(grouped, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(plot_metric_bars(results_df, "AvgProfit"))
                    st.pyplot(plot_metric_bars(results_df, "SurvivalRate"))
                with col2:
                    st.pyplot(plot_metric_bars(results_df, "AvgDependency"))
                    st.pyplot(plot_metric_bars(results_df, "TotalEmissions"))

                st.write("### Trade-off-kuva")
                st.pyplot(plot_tradeoff(results_df))

                with st.expander("Näytä raakadata"):
                    st.dataframe(results_df, use_container_width=True)
