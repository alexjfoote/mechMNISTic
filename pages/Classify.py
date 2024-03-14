import streamlit as st
import json
import matplotlib as mpl
import functools
import torch
import numpy as np

from skimage.transform import resize
from streamlit_drawable_canvas import st_canvas

import display

from Home import (
    model,
    examples,
    test_examples,
    weights,
    output_weights
)
from pages.Neurons import show_neuron
from model import center_transform

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

if "stroke_width" not in st.session_state:
    st.session_state["stroke_width"] = 20

# ============== FUNCTIONALITY ==============

def classification(draw):
    input_container = st.empty()

    image_width = 0.35
    logit_width = 0.13
    other_width = 1 - (2 * logit_width)
    image_col, logit_col, prob_col, other_col = st.columns([image_width, logit_width, logit_width, other_width])

    if draw:
        with image_col:
            st.subheader("Input")
            st.caption("Draw a digit to classify")
            canvas_result = st_canvas(
                stroke_width=st.session_state["stroke_width"],
                stroke_color="rgba(255, 255, 255, 1)",
                background_color="rgba(0, 0, 0, 1)",
                update_streamlit=True,
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            st.slider("Stroke Width", 5, 50, st.session_state["stroke_width"], key="stroke_width")
        if canvas_result.image_data is None:
            return
        scaled = resize(canvas_result.image_data, (28, 28), anti_aliasing=True)
        grayscale = scaled[:, :, 1]
        example = center_transform(grayscale.astype(np.float32)).unsqueeze(0)
    else:
        with input_container:
            input_width = 0.25
            input_col, _ = st.columns([input_width, 1 - input_width])
            with input_col:
                example_idx = st.number_input(
                    "Example Index", min_value=0, max_value=len(test_examples) - 1, value=0,
                    help=f"Choose an example to classify (0 to {len(test_examples) - 1})"
                )

        with image_col:
            example = test_examples[example_idx][0].unsqueeze(0)
            st.subheader("Input")
            st.caption("Example from the test set")
            plot(display.raw(example, colorbar=False))

    _, layer_activations = model(example)

    activations = layer_activations[1][0]

    logit_contributions = output_weights[:, list(range(64))] * activations
    logits = torch.sum(logit_contributions, dim=1)
    probabilities = torch.softmax(logits, dim=0)

    max_abs_contribution = display.max_absolute(logit_contributions)

    with logit_col:
        st.subheader("Prediction")
        st.caption("Logits")
        plot(display.raw(
            logits.unsqueeze(0).t(), square=False, axis=True, cmap="coolwarm", figsize=(1, 1.7),
            yt=[i for i in range(len(logits))], colorbar=False
        ))
    with prob_col:
        st.subheader("\u200B")
        st.caption("Probabilities")
        plot(display.raw(
            probabilities.unsqueeze(0).t(), square=False, axis=True, figsize=(1, 1.7),
            yt=[i for i in range(len(logits))], colorbar=False
        ))

    neuron_ticks = [i for i in range(len(activations)) if i % 5 == 0]

    st.subheader("Output Weights")
    st.caption("Weights connecting each neuron to the output layer")
    plot(display.output_weights(
        output_weights, list(range(64)), xt=neuron_ticks, colorbar_kwargs={"pad": 0.003}
    ))

    st.subheader("Activations")
    st.caption("Activation of each neuron in the hidden layer")
    pad_width = 0.015
    activation_width = 0.945
    _, activation_col, _ = st.columns([pad_width, activation_width, 1 - activation_width - pad_width])
    with activation_col:
        plot(display.raw(
            activations, square=False, figsize=((len(activations) / 3), 2), axis=True,
            xt=neuron_ticks, colorbar_kwargs={"orientation": "vertical", "pad": 0.01}
        ))

    st.subheader("Logit Effects")
    st.caption("Activations ⊙ Output Weights - the contribution of each neuron to the logit of each class")
    plot(display.raw(
        logit_contributions, square=False, figsize=(len(activations) / 2, 2), axis=True, cmap="coolwarm",
        vmin=-max_abs_contribution, vmax=max_abs_contribution,
        xt=neuron_ticks, yt=[i for i in range(len(output_weights))], colorbar_kwargs={"pad": 0.01}
    ))

    threshold = 0.025

    st.subheader("Important Neurons")
    st.caption(f"Neurons that contributed more than {threshold:.1%} to the logit of the selected class")

    prediction = torch.argmax(probabilities).item()

    width = 0.1
    width_2 = 0.6
    col_1, col_2, col_3 = st.columns([width, width_2, 1 - width - width_2])
    with col_1:
        st.caption("Select a class")
        logit_to_view = st.selectbox("Class", list(range(10)), index=prediction)

    logit_row = logit_contributions[logit_to_view]
    neuron_contributions = torch.abs(logit_row)
    total_logit = torch.sum(neuron_contributions)
    important_neurons = torch.nonzero(neuron_contributions > total_logit * threshold).squeeze(1)
    important_contributions = output_weights[logit_to_view, important_neurons] * activations[important_neurons]

    st.write(f"**Logit: {logits[logit_to_view]:.2f}, Probability: {probabilities[logit_to_view]:.2f}**")
    with col_2:
        st.caption("Logit Effects")
        plot(display.raw(
            important_contributions, square=False, axis=True, cmap="coolwarm",
            vmin=-max_abs_contribution, vmax=max_abs_contribution,
            figsize=(len(important_contributions) / 2, 2),
            # figsize=(2, len(important_contributions)),
            xt=([i for i in range(len(important_contributions))], important_neurons.tolist()),
            yt=[], colorbar_kwargs={"pad": 0.1, "orientation": "horizontal"}, colorbar=False
        ))

    max_cols = 8
    n_cols = max(min(len(important_neurons), max_cols), 5)
    cols = st.columns(n_cols)

    combined_embedding = torch.zeros_like(example)

    for i, neuron in enumerate(important_neurons):
        kernel = weights[0, neuron, :]
        feature_embedding = example * kernel
        combined_embedding += feature_embedding * output_weights[logit_to_view, neuron]
        col = cols[i % n_cols]
        with col:
            color = "blue" if important_contributions[i] < 0 else "orange"
            st.subheader(f"Neuron {neuron}")
            st.write(f"Activation: {activations[neuron]:.1f}")
            st.write(f"Weight: {output_weights[logit_to_view, neuron]:.1f}")
            st.write(f":{color}[**Logit Effect: {important_contributions[i]:.1f}**]")
            plot(display.kernel(weights, 0, neuron, colorbar=False, cmap="gray"))
            plot(display.feature_embedding(example, weights, 0, neuron))

    max_effect_index = torch.argmax(torch.abs(important_contributions)).item()
    width = 0.1
    col_1, _ = st.columns([width, 1 - width])
    with col_1:
        neuron_to_view = st.selectbox("Neuron to View", [e.item() for e in important_neurons], index=max_effect_index)

    st.subheader(f"Neuron {neuron_to_view}")
    show_neuron(0, neuron_to_view)


# ============== PAGE ==============

st.title("Understanding Classification")
draw = st.toggle("DIY", help="Draw your own example or choose from the test set", value=True)

classification(draw)


