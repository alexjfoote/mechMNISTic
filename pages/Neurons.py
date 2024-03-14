import streamlit as st
import json
import matplotlib as mpl
import functools

import display

from Home import (
    statistics,
    clusters,
    examples,
    weights,
    output_weights,
)

# ============== CONFIG ==============

model_name = "1_layer_64_neurons"
model_path = f"models/{model_name}"

with open(f"{model_path}/config.json") as ifh:
    config = json.load(ifh)

layers = config["layers"]
neurons = config["neurons"]

max_clusters = 10

mpl.rcParams['figure.max_open_warning'] = 0

plot = functools.partial(st.pyplot, clear_figure=True)

# ============== FUNCTIONALITY ==============


def show_neuron(layer, neuron, max_clusters=max_clusters):
    width_1 = 0.4
    width_2 = 0.25
    col_1, col_2, col_3 = st.columns([width_1, width_2, width_1])

    with col_1:
        st.subheader(f"Input Weights")
        st.caption("Input weights visualised as a 28x28 image")
        st.caption("Each weight connects to a pixel in the input image")
        plot(display.kernel(weights, layer, neuron))
    with col_2:
        st.subheader(f"Output Weights")
        st.caption("Each weight connects to an output class")
        st.caption("\u200B")
        plot(display.output_weights(output_weights, neuron, colorbar_kwargs={"pad": 0.2}))
    with col_3:
        st.subheader(f"Activation Distribution")
        st.caption("Cumulative distribution of activation values")
        st.caption("\u200B")
        plot(display.activation_statistic(statistics, layer, neuron))

    # TODO - sort by max activation for clusters with 3+ examples
    cluster_figs, central_idxs = display.cluster(
        examples, clusters, layer, neuron, max_per_cluster=3, max_clusters=max_clusters
    )

    cluster_width = 0.6
    other_width = (1 - cluster_width) / 2
    for i, (fig, central_idx) in enumerate(zip(cluster_figs, central_idxs)):
        col_1, col_2, col_3 = st.columns([cluster_width, other_width, other_width])
        with col_1:
            if i == 0:
                st.subheader(f"Top {max_clusters} Feature Clusters")
                st.caption("Ordered by number of elements, clustered by feature similarity")
            plot(fig)
        with col_2:
            if i == 0:
                st.subheader(f"Weights")
                st.caption("Learned Input Weights")
            plot(display.kernel(weights, layer, neuron, colorbar=False, cmap="gray"))
        with col_3:
            if i == 0:
                st.subheader(f"Feature")
                st.caption("Central cluster example ⊙ Weights")
            plot(display.feature_embedding(examples[central_idx][0], weights, layer, neuron))


# ============== PAGE ==============

if __name__ == "__main__":
    st.title("Understanding Neurons")
    layer = 0
    neuron = st.number_input(f"Select Neuron (0 to {neurons - 1})", min_value=0, max_value=neurons, value=0, help="Choose a neuron to view")

    show_neuron(layer, neuron)
