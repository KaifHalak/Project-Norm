"""Streamlit dashboard for the artificial-life simulation.

Run with: ``streamlit run app.py``
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from simulation import SimulationController


st.set_page_config(page_title="Norn Simulation", layout="wide")


def get_controller():
    if "simulation" not in st.session_state:
        st.session_state.simulation = SimulationController()
    return st.session_state.simulation


def history_frame(controller):
    return pd.DataFrame(controller.history)


def main():
    controller = get_controller()
    st.title("Norn Artificial-Life Simulation")

    with st.sidebar:
        st.header("Simulation controls")
        population_size = st.number_input(
            "Population size", min_value=2, value=controller.population_size
        )
        ticks_per_generation = st.number_input(
            "Ticks per generation", min_value=1,
            value=controller.ticks_per_generation
        )
        mutation_rate = st.slider(
            "Mutation rate", 0.0, 1.0, controller.mutation_rate, 0.01
        )
        if st.button("Initialize / reset", use_container_width=True):
            st.session_state.simulation = SimulationController(
                population_size=int(population_size),
                ticks_per_generation=int(ticks_per_generation),
                mutation_rate=mutation_rate,
            )
            st.rerun()

        st.divider()
        st.write(
            f"Tick **{controller.total_elapsed_ticks}** | "
            f"Generation **{controller.generation}**"
        )
        st.write(
            f"Population: **{len(controller.population.norns)}** | "
            f"Living: **{len(controller.population.get_alive_norns())}**"
        )
        columns = st.columns(4)
        for column, amount in zip(columns, (1, 10, 50, None)):
            label = "Custom" if amount is None else f"+{amount}"
            if amount is None:
                amount = st.number_input(
                    "Custom ticks", min_value=0, value=5, step=1
                )
            if column.button(label, use_container_width=True):
                controller.advance_ticks(int(amount))
                st.rerun()

    tabs = st.tabs([
        "Overview", "Individual Norn", "Neural Network",
        "Genetics", "Evolution"
    ])
    with tabs[0]:
        render_overview(controller)
    with tabs[1]:
        render_individual(controller)
    with tabs[2]:
        render_network(controller)
    with tabs[3]:
        render_genetics(controller)
    with tabs[4]:
        render_evolution(controller)


def render_overview(controller):
    norns = controller.population.norns
    alive = controller.population.get_alive_norns()
    metrics = st.columns(4)
    metrics[0].metric("Living Norns", f"{len(alive)}/{len(norns)}")
    metrics[1].metric(
        "Average health",
        f"{sum(n.health for n in norns) / len(norns):.3f}" if norns else "0",
    )
    metrics[2].metric(
        "Average fitness",
        f"{sum(n.get_fitness() for n in norns) / len(norns):.2f}"
        if norns else "0",
    )
    metrics[3].metric("Status", "Extinct" if controller.extinct else "Running")
    frame = history_frame(controller)
    if not frame.empty:
        st.plotly_chart(
            px.line(frame, x="tick", y=["alive", "average_health"],
                    title="Population health over time"),
            use_container_width=True,
        )
    st.dataframe(pd.DataFrame([
        {"Norn": i, "Alive": n.alive, "Age": n.age,
         "Health": n.health, "Fitness": n.get_fitness()}
        for i, n in enumerate(norns, 1)
    ]), use_container_width=True)


def render_individual(controller):
    norns = controller.population.norns
    if not norns:
        st.info("No Norns are available.")
        return
    selected = st.selectbox("Norn", range(1, len(norns) + 1))
    norn = norns[selected - 1]
    st.write({
        "health": norn.health, "hunger": norn.body.get_chemical("hunger"),
        "pain": norn.body.get_chemical("pain"), "age": norn.age,
        "fitness": norn.get_fitness(), "alive": norn.alive,
    })
    rows = []
    for snapshot in controller.history:
        for item in snapshot["norns"]:
            if item["norn_id"] == selected:
                row = {"tick": snapshot["tick"], **item["chemicals"],
                       "health": item["health"], "fitness": item["fitness"]}
                rows.append(row)
    if rows:
        st.plotly_chart(
            px.line(pd.DataFrame(rows), x="tick",
                    y=["hunger", "pain", "health"], title="Body state"),
            use_container_width=True,
        )
    st.write("Recent actions")
    events = [
        {"tick": tick["tick"], **event}
        for tick in controller.history
        for event in tick["events"]
        if event["norn_id"] == selected
    ]
    st.dataframe(pd.DataFrame(events).tail(20), use_container_width=True)


def render_network(controller):
    norns = controller.population.norns
    if not norns:
        return
    selected = st.selectbox("Network Norn", range(1, len(norns) + 1),
                            key="network_norn")
    snapshot = controller._norn_snapshot(selected, norns[selected - 1])
    st.dataframe(pd.DataFrame([
        {"neuron": name, "state": state}
        for name, state in snapshot["neurons"].items()
    ]), use_container_width=True)
    st.dataframe(pd.DataFrame(snapshot["synapses"]), use_container_width=True)
    rows = []
    for tick in controller.history:
        item = next((x for x in tick["norns"]
                     if x["norn_id"] == selected), None)
        if item:
            for connection in item["synapses"]:
                rows.append({"tick": tick["tick"],
                             "connection": f"{connection['source']} -> "
                                           f"{connection['destination']}",
                             "stw": connection["stw"], "ltw": connection["ltw"]})
    if rows:
        st.plotly_chart(
            px.line(pd.DataFrame(rows), x="tick", y=["stw", "ltw"],
                    color="connection", title="Synaptic learning"),
            use_container_width=True,
        )


def render_genetics(controller):
    norns = controller.population.norns
    if not norns:
        return
    data = pd.DataFrame([
        {"Norn": i, **norn.genome.genes}
        for i, norn in enumerate(norns, 1)
    ])
    st.dataframe(data, use_container_width=True)
    selected = st.multiselect("Compare Norns", list(data["Norn"]),
                              default=list(data["Norn"])[:2])
    if selected:
        st.plotly_chart(
            px.bar(data[data["Norn"].isin(selected)].melt(id_vars="Norn"),
                   x="variable", y="value", color="Norn",
                   title="Inherited traits"),
            use_container_width=True,
        )


def render_evolution(controller):
    frame = pd.DataFrame(controller.generation_history)
    if frame.empty:
        st.info("Complete a generation to see evolution statistics.")
        return
    st.plotly_chart(
        px.line(frame, x="generation",
                y=["average_fitness", "best_fitness"],
                markers=True, title="Fitness by generation"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.line(frame, x="generation",
                y=["alive", "average_learning_rate", "average_hunger_rate"],
                markers=True, title="Survival and genetic traits"),
        use_container_width=True,
    )
    st.dataframe(frame, use_container_width=True)


if __name__ == "__main__":
    main()
