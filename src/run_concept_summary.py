import json
import os
import pickle
import random
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
import umap
from sklearn import preprocessing
from sklearn.cluster import KMeans


def set_up_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def find_elbow_k(X, K, fax):
    """Finds the optimal number of clusters using the Elbow Method.

    Args:
        X (np.ndarray): Feature matrix.
        K (iterable): List or range of k values to test.
        plot (bool): Whether to plot the elbow chart.

    Returns:
        int: Estimated optimal k based on max curvature.
    """
    inertias = []

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    inertias = np.array(inertias)

    # Compute 1st and 2nd differences
    delta = np.diff(inertias)
    delta2 = np.diff(delta)

    if len(delta2) == 0:
        raise ValueError("Need at least 3 values in K to compute elbow point.")

    elbow_index = np.argmax(-delta2) + 2  # +2 to correct indexing

    fig, ax = fax
    ax.plot(K, inertias, "bo-", label="Inertia")
    ax.axvline(
        K[elbow_index - 1],
        color="red",
        linestyle="--",
        label=f"Elbow at k={K[elbow_index - 1]}",
    )
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method For Optimal k")
    ax.grid(True)
    ax.legend()


def _recover_control_sequences(s: str) -> str:
    s = s.replace(r"\}", "}")
    s = s.replace(r"\{", "{")
    s = s.replace(r"textbackslash ", "")
    s = s.replace(r"\end{tabular} \\", r"\end{tabular} \\ \midrule")
    return s


def get_colordict(k):
    random.seed(0)
    return {i: (random.random(), random.random(), random.random()) for i in range(k)}


def get_summary_closedai(content, model_labeling):
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    annotation_request = [
        {
            "role": "user",
            "content": f"Please assign a concise metalabel that accurately reflects the overall content of the following text descriptions. If the texts can not be summarized meaningfully, return 'None': {content} ",
        },
    ]

    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }
    content = model_labeling.generate_content(
        annotation_request[0]["content"], safety_settings=safety_settings
    )
    response = content.text

    return response


def get_summary(pipeline, tokenizer, content):
    annotation_request = [
        {
            "role": "user",
            "content": f"Please assign a concise metalabel that accurately reflects the overall content of the following text descriptions. If the texts can not be summarized meaningfully, return 'None': {content} ",
        },
    ]

    # Define the phrase you want to avoid
    bad_phrase = "The"

    # Encode the phrase to get the token IDs
    bad_phrase_ids = tokenizer.encode(bad_phrase, add_special_tokens=False)

    responses = pipeline(
        annotation_request,
        do_sample=False,
        num_return_sequences=1,
        return_full_text=False,
        max_new_tokens=30,
        temperature=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        repetition_penalty=1,
        bad_words_ids=[bad_phrase_ids],
    )

    response = responses[0]["generated_text"].strip()

    return response


model_case = "gpt2-small-sae"  # "gpt2-xl"

src = os.path.join("../descriptions/gemini-1-5-pro/", model_case)

exp_dir = "metalabels"
set_up_dir(exp_dir)

exp_dir = os.path.join(f"../{exp_dir}", f"{model_case}")
set_up_dir(exp_dir)

# Model to extract metalabels
label_model = "gemini-1-5-pro"

# Model to get sentence representations for the feature descriptions (used for clustering)
embed_model_name = "gpt-xl"

# List of clusters to check for elbow
Ks = range(25, 200, 25)


csv_files = [f for f in os.listdir(src) if f.endswith(".csv")]
dfs = {}
for csv_file in csv_files:
    file_path = os.path.join(src, csv_file)
    file_name = os.path.split(file_path)[1]

    match = re.search(r"layer-(\d+)_unit-(\d+)", file_name)
    layer = str(match.group(1))
    unit = str(match.group(2))
    df = pd.read_csv(file_path)

    if layer not in dfs:
        dfs[layer] = {}
    dfs[layer][unit] = df


# Labeling
from transformers import AutoModelForCausalLM, AutoTokenizer

if embed_model_name == "gpt-xl" or embed_model_name == "gpt2-small-sae":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    _ = model.eval()
    _ = model.cuda()
else:
    pass


for embed_case in ["final_token"]:
    data_all = []
    for k in sorted(dfs.keys()):
        df = dfs[k]
        for unit in sorted(df.keys()):
            texts = df[unit].description
            for i, t in enumerate(texts):
                with torch.no_grad():
                    inputs = tokenizer(t, return_tensors="pt").to("cuda")
                    # Get embeddings
                    outputs = model(**inputs, output_hidden_states=True)

                    if embed_case == "tokens":
                        token_embeddings = outputs["hidden_states"][0]
                        x = token_embeddings.detach().cpu().numpy().mean(1).squeeze()
                    elif embed_case == "last":
                        last_layer_embeddings = outputs["hidden_states"][-1]
                        x = (
                            last_layer_embeddings.detach()
                            .cpu()
                            .numpy()
                            .mean(1)
                            .squeeze()
                        )
                    elif embed_case == "final_token":
                        last_layer_embeddings = outputs["hidden_states"][-1]
                        x = (
                            last_layer_embeddings.detach()
                            .cpu()
                            .numpy()[:, -1]
                            .squeeze()
                        )
                    else:
                        raise

                d = [x, t, k, unit]
                data_all.append(d)

    if label_model == "llama33_70B":
        tokenizer_labeling = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.3-70B-Instruct"
        )
        model_labeling = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.3-70B-Instruct",
            evice_map="auto",
            torch_dtype=torch.float16,
        )

    elif label_model == "llama31_8B":
        tokenizer_labeling = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct"
        )
        model_labeling = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
        )

    elif label_model == "llama31_70B":
        tokenizer_labeling = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-70B-Instruct"
        )
        model_labeling = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-70B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
        )

    elif label_model == "gemini-1-5-pro":
        import google.generativeai as genai

        gemini_key = os.environ["GEMINI_KEY"]
        genai.configure(api_key=gemini_key)
        model_labeling = genai.GenerativeModel("gemini-1.5-pro")
        tokenizer_labeling = None

    else:
        raise

    if label_model != "gemini-1-5-pro":
        _ = model_labeling.eval()

    out_dir_src = os.path.join(exp_dir, embed_model_name + "_" + embed_case)

    X = np.array([x[0] for x in data_all])
    T = np.array([x[1] for x in data_all])
    Ls = np.array([x[2] for x in data_all])
    Units = np.array([x[3] for x in data_all])

    print(X.shape)

    # Normalize input data
    X = np.array([x[0] for x in data_all])
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    # Normalize input data
    X = X / np.linalg.norm(X, axis=1, keepdims=1)
    # UMAP embedding

    print(np.linalg.norm(X, axis=1, keepdims=1).shape)

    # KMeans clustering
    fig, ax = plt.subplots(figsize=(8, 5))

    find_elbow_k(X, Ks, (fig, ax))
    fig.savefig(os.path.join(exp_dir, "elbow.pdf"), dpi=300)

    plot_data = {}
    for n_clusters in Ks:
        n_plot = n_clusters

        out_dir = out_dir_src + "_" + str(n_clusters)
        set_up_dir(out_dir)

        clustering = KMeans(n_clusters=n_clusters, random_state=42)

        colordict = get_colordict(n_clusters)

        cluster_labels = clustering.fit_predict(X)

        from sklearn.metrics import pairwise_distances

        cluster_dict = {"variances": {}, "texts": {}}
        for cluster_id in range(n_clusters):
            idxs = np.where(cluster_labels == cluster_id)[0]
            points = X[idxs]  # original high-dimensional points

            cluster_dict["texts"][cluster_id] = T[idxs]

            if True:
                centroid = clustering.cluster_centers_[cluster_id]

                # Squared distances from each point to the centroid
                dists = pairwise_distances(points, [centroid], metric="euclidean") ** 2
                variance = np.mean(dists)
                cluster_dict["variances"][cluster_id] = variance

        cluster_variances = cluster_dict["variances"]
        sorted_cluster_ids = sorted(cluster_variances, key=cluster_variances.get)
        sorted_variances = [cluster_variances[id] for id in sorted_cluster_ids]

        plt.plot(sorted(cluster_variances.values()))
        plt.show()

        if label_model != "gemini-1-5-pro":
            labeling_pipeline = transformers.pipeline(
                "text-generation",
                model=model_labeling,
                tokenizer=tokenizer_labeling,
            )

        structured_output = {}
        meta_labels = {id: {} for id in sorted_cluster_ids[:n_plot]}

        for id in sorted_cluster_ids[:n_plot]:
            content = ",\n".join(cluster_dict["texts"][id])
            if label_model == "gemini-1-5-pro":
                label = get_summary_closedai(content, model_labeling)
            else:
                label = get_summary(labeling_pipeline, tokenizer_labeling, content)

            # remove quotation marks
            label = re.sub(r'^"|"$', "", label)

            print(f"***{id}***")
            print("\n".join(cluster_dict["texts"][id]))
            print("\n")

            print(label)

            meta_labels[id] = label.strip()

            structured_output[id] = {
                "label": label,
                "texts": cluster_dict["texts"][id].tolist(),
            }

            print("\n\n")

        # Write to a JSON file
        with open(
            os.path.join(out_dir, f"cluster_labels_{label_model}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(structured_output, f, indent=2, ensure_ascii=False)

        project = umap.UMAP(
            n_components=2,
            n_neighbors=10,
            min_dist=0.1,
            random_state=42,
            metric="euclidean",
        )
        X_2d = project.fit_transform(X)

        # Plot setup
        cmap = matplotlib.colormaps["viridis"]
        f, ax = plt.subplots(1, 1, figsize=(7, 8))

        # Plot clusters
        for cluster_id in range(n_clusters):
            idxs = np.where(cluster_labels == cluster_id)[0]
            points = X_2d[idxs]

            # Cluster color
            cluster_color = colordict[cluster_id]  # cmap(cluster_id / n_clusters)
            ax.scatter(points[:, 0], points[:, 1], color=cluster_color, s=50, alpha=0.9)

            if False:
                # Optionally annotate
                for i in idxs:
                    label_short = " ".join(T[i].split()[:4])
                    ax.annotate(label_short, (X_2d[i, 0], X_2d[i, 1]), fontsize=2)

            if cluster_id in sorted_cluster_ids[:n_plot]:
                x_mean, y_mean = np.mean(points[:, 0]), np.mean(points[:, 1])
                label_summary = meta_labels[cluster_id]
                ax.annotate(
                    label_summary,
                    (x_mean, y_mean),
                    fontsize=8,
                    c=cluster_color,
                    ha="center",
                    va="center",
                )

        ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
        ax.set_axis_off()
        f.tight_layout()
        f.savefig(
            os.path.join(out_dir, f"2D_{embed_case}_labels_{label_model}.pdf"), dpi=300
        )
        plt.show()

        # Convert to list of rows
        rows = []
        for cluster_id, info in structured_output.items():
            label = info["label"]
            sample_texts = [t.strip().replace("\\", "") for t in info["texts"]]
            sample_text = "\n".join(sample_texts)

            formatted_texts = (
                r"\begin{tabular}[t]{@{}p{9cm}}"
                + r" \\ ".join(
                    t.replace("\n", " ").replace("_", r"\_") for t in sample_texts
                )
                + r"\end{tabular}"
            )

            rows.append(
                {"cluster_id": cluster_id, "label": label, "texts": formatted_texts}
            )

        # Create DataFrame
        df = pd.DataFrame(rows)

        str_ = df.to_latex(index=False)

        latex_table = _recover_control_sequences(str_)

        with open(
            os.path.join(out_dir, "cluster_table.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(latex_table)

        plot_data = {
            "X": X,
            "X_2d": X_2d,
            "sorted_cluster_ids": sorted_cluster_ids,
            "T": T,
            "cluster_labels": cluster_labels,
            "meta_labels": meta_labels,
            "data_all": data_all,
            "cluster_dict": cluster_dict,
        }
        with open(os.path.join(out_dir, "plot_data.pkl"), "wb") as fp:
            pickle.dump(plot_data, fp)
