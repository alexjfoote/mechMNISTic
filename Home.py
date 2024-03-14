import streamlit as st
import json
import torch
import torch.nn.functional as F

from torchvision import datasets

import utils

from model import MLP, transform

# ============== CONFIG ==============

st.set_page_config(page_title="MechMNISTic", layout="wide")

model_name = "1_layer_64_neurons"
model_path = f"models/{model_name}"

# ============== DATA ==============


@st.cache_resource
def load():
    with open(f"{model_path}/activation_statistics.json") as ifh:
        statistics = utils.to_dict(json.load(ifh))

    with open(f"{model_path}/clusters.json") as ifh:
        clusters = utils.to_dict(json.load(ifh))

    with open(f"{model_path}/indexing.json") as ifh:
        indexing = json.load(ifh)

    examples = datasets.MNIST(f'data', train=True, download=True, transform=transform)
    test_examples = datasets.MNIST(f'data', train=False, download=True, transform=transform)

    model = MLP(inference=True)
    model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=torch.device('cpu')))
    model.eval()

    full_weights = []

    for i, (name, parameters) in enumerate(model.named_parameters()):
        if "weight" not in name:
            continue

        full_weights.append(parameters)

    weights = torch.stack(full_weights[:-1])
    output_weights = full_weights[-1]

    return statistics, clusters, indexing, examples, test_examples, model, weights, output_weights


statistics, clusters, indexing, examples, test_examples, model, weights, output_weights = load()

# ============== PAGE ==============

if __name__ == "__main__":
    with open("README.md") as ifh:
        readme = "\n".join(ifh.readlines())

    st.write(readme)
