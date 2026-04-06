import pandas as pd
import streamlit as st

from app_utils import (
    MODEL_SPECS,
    GraphProjectPredictor,
    build_experiment_summary,
    get_device,
    get_label_distribution,
    get_samples_for_label,
    load_architecture_images,
    load_dataset,
    load_pil_image,
    load_training_log,
)


st.set_page_config(
    page_title="Graph Networks Explorer",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(64, 84, 117, 0.28), transparent 28%),
            radial-gradient(circle at top right, rgba(153, 76, 61, 0.18), transparent 24%),
            linear-gradient(180deg, #0d1117 0%, #111827 100%);
        color: #f3f4f6;
    }
    [data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    .hero {
        padding: 0.2rem 0 0.9rem 0;
        color: #f9fafb;
        margin-bottom: 0.6rem;
    }
    .hero h1 {
        margin: 0 0 0.25rem 0;
        font-size: 2.2rem;
        line-height: 1.1;
    }
    .hero p {
        margin: 0;
        font-size: 1.05rem;
        max-width: 760px;
    }
    .small-note {
        color: #94a3b8;
        font-size: 1rem;
    }
    .compact-list {
        margin: 0;
        padding-left: 1rem;
        color: #d1d5db;
    }
    .compact-list li {
        margin: 0.2rem 0;
    }
    h3 {
        margin-top: 0.6rem;
        font-size: 1.45rem;
    }
    h1, h2, h3, h4, p, li, label, div {
        font-size: 1.05rem;
    }
    .stMarkdown p, .stMarkdown li {
        font-size: 1.08rem;
        line-height: 1.8;
    }
    .stCaption {
        font-size: 0.98rem;
    }
    pre {
        background: rgba(2, 6, 23, 0.95) !important;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 0.9rem !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.9) !important;
        color: #f9fafb !important;
        font-size: 1rem !important;
    }
    .stButton button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
    }
    .paper-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-top: 0.2rem;
    }
    .paper-keywords {
        color: #93c5fd;
        font-size: 0.98rem;
        margin-top: 0.45rem;
    }
    .paper-links {
        color: #cbd5e1;
        font-size: 1rem;
        margin-top: 0.45rem;
        display: flex;
        gap: 1rem;
        align-items: center;
        flex-wrap: wrap;
    }
    .paper-links a {
        color: #93c5fd;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
    }
    .paper-links a:hover {
        text-decoration: underline;
    }
    .social-icon {
        width: 1rem;
        height: 1rem;
        display: inline-block;
    }
    .article-note {
        color: #d1d5db;
        line-height: 1.75;
        font-size: 1.12rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_data(show_spinner=False)
def get_experiment_table() -> pd.DataFrame:
    return build_experiment_summary()


@st.cache_data(show_spinner=False)
def get_log_df(model_key: str) -> pd.DataFrame:
    return load_training_log(model_key)


@st.cache_resource(show_spinner=False)
def get_predictor(model_key: str, device: str) -> GraphProjectPredictor:
    return GraphProjectPredictor(model_key=model_key, device=device)


def render_overview(dataset_df: pd.DataFrame, experiment_df: pd.DataFrame) -> None:
    best_row = experiment_df.iloc[0]
    graph_row = experiment_df.loc[experiment_df["model_key"] == "text_image_graph"].iloc[0]

    st.markdown(
        """
        <div class="hero">
            <h1>Multi-Graph Contextual Learning for Multimodal Meme Classification</h1>
            <div class="paper-subtitle">
                Vishnu Chityala - e23cseu0049@bennett.edu.in,
                Harshil Uttaradhi - e23cseu0073@bennett.edu.in,
                Sutejas Hashia - e23cseu0047@bennett.edu.in
            </div>
            <div class="paper-keywords">Keywords: Multimodal Meme Classification, Graph Neural Networks, Graph Attention Networks, Multimodal Learning</div>
            <div class="paper-links">
                <a href="https://www.linkedin.com/in/vishnuchityala/" target="_blank">
                    <svg class="social-icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                        <path d="M6.94 8.5H3.56V20h3.38V8.5ZM5.25 3A2 2 0 1 0 5.3 7a2 2 0 0 0-.05-4ZM20.44 12.87c0-3.47-1.85-5.08-4.31-5.08a3.76 3.76 0 0 0-3.39 1.87V8.5H9.36c.04.77 0 11.5 0 11.5h3.38v-6.42c0-.34.02-.68.13-.92a2.22 2.22 0 0 1 2.08-1.48c1.47 0 2.06 1.12 2.06 2.76V20h3.38v-7.13Z"/>
                    </svg>
                    LinkedIn
                </a>
                <a href="https://github.com/vishnurchityala/graph-networks" target="_blank">
                    <svg class="social-icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                        <path d="M12 .5a12 12 0 0 0-3.79 23.39c.6.11.82-.26.82-.58v-2.03c-3.34.73-4.04-1.41-4.04-1.41-.55-1.38-1.33-1.75-1.33-1.75-1.09-.75.08-.73.08-.73 1.2.08 1.84 1.24 1.84 1.24 1.08 1.83 2.82 1.3 3.5 1 .11-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.23-3.22-.12-.3-.53-1.52.12-3.17 0 0 1-.32 3.3 1.23A11.3 11.3 0 0 1 12 6.57c1.02 0 2.04.14 3 .4 2.3-1.55 3.3-1.23 3.3-1.23.65 1.65.24 2.87.12 3.17.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.62-5.48 5.92.43.37.81 1.09.81 2.2v3.26c0 .32.21.69.82.58A12 12 0 0 0 12 .5Z"/>
                    </svg>
                    GitHub
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Abstract")
    st.markdown(
        f"""
        <div class="article-note">
        This work proposes a multi-graph contextual classification framework for multimodal
        meme analysis that leverages relationships between samples rather than treating them
        independently. Experiments are conducted on the What’s Beneath Misogynous
        Stereotyping (WBMS) Dataset, which contains meme images and captions annotated for
        misogynistic content. Text and image features are extracted using BERT and CLIP,
        respectively, and reduced using Principal Component Analysis (PCA). Three similarity
        graphs are then constructed using cosine similarity-based k-nearest neighbors (kNN):
        a text semantic graph, a visual semantic graph, and a class-discriminative graph
        obtained using Linear Discriminant Analysis (LDA). These graphs are processed using
        Graph Attention Networks (GAT) to learn contextual node representations through
        attention-based neighbor aggregation. The contextual features from the graphs are fused
        to perform final classification. This approach aims to incorporate dataset-level
        contextual information by modeling statistical relationships between samples in the
        embedding space.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Introduction")
    st.markdown(
        """
        <div class="article-note">
        The proposed method introduces a multi-graph contextual classification framework that
        models relationships between samples in a multimodal dataset. Instead of predicting from
        independent features, the dataset is represented as a set of similarity graphs
        constructed using cosine similarity-based k-nearest neighbor connections. Three
        complementary graphs are constructed: a text semantic graph, a visual semantic graph,
        and a class-discriminative graph. Each graph captures a different statistical structure
        of the dataset. The graphs are processed using Graph Attention Networks to learn
        contextual node representations, and the resulting features are fused for final
        classification.
        </div>
        """,
        unsafe_allow_html=True,
    )

    images = load_architecture_images()
    if images:
        st.image(str(images[0]["path"]), caption="Figure: Model overview", use_container_width=True)

    st.markdown("### Method")
    st.markdown(
        """
        <div class="article-note">
        Each data sample contains textual and visual components. Text captions are encoded
        using BERT to obtain semantic text embeddings, while images are encoded using CLIP to
        generate visual feature representations. Since these embeddings are high-dimensional,
        Principal Component Analysis is applied separately to text and image features to obtain
        compact representations suitable for graph construction.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Code snippet: multimodal feature extraction")
    st.code(
        """text_features = self.text_embedder(text_inputs)
image_features = self.image_embedder(image_inputs)
combined_embed = torch.cat([text_features, image_features], dim=1)""",
        language="python",
    )

    st.markdown("### Graph Construction")
    st.markdown(
        """
        <div class="article-note">
        Three graphs are constructed to model different forms of relationships among samples.
        The text semantic graph captures linguistic relationships between captions using cosine
        similarity between PCA-reduced BERT embeddings. The visual semantic graph represents
        similarity between images using cosine similarity over PCA-reduced CLIP image
        embeddings. To model class-level statistical relationships, text and image PCA features
        are combined to form a multimodal representation. Linear Discriminant Analysis is then
        applied to project the data into a space that maximizes class separability, and a kNN
        graph is constructed in this LDA space.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Code snippet: graph-based context")
    st.code(
        """lda_features = self.lda_layer(combined_embed)
lda_graph_features = self.graph_module(lda_features)
text_graph_features = self.graph_text_module(text_features)
image_graph_features = self.graph_image_module(image_features)""",
        language="python",
    )

    st.markdown("### Contextual Representation Learning")
    st.markdown(
        """
        <div class="article-note">
        Each graph is processed using a Graph Attention Network, which allows nodes to
        aggregate information from neighboring samples while learning attention weights that
        determine the importance of each neighbor. This enables the model to refine node
        representations using contextual information from similar samples.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Dataset")
    st.markdown(
        """
        <div class="article-note">
        The project is built around meme images and captions from the WBMS setting. In this app,
        the active 4-class version focuses on the stereotype groups used in the local dataset:
        Kitchen, Working, Leadership, and Shopping. The current local setup contains
        <strong>"""
        + f"{len(dataset_df):,}"
        + """</strong> samples across these four classes.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        - Kitchen
        - Working
        - Leadership
        - Shopping
        """
    )
    distribution_df = get_label_distribution(dataset_df).set_index("label")
    st.bar_chart(distribution_df)
    st.caption("Class distribution in the local 4-class dataset used by this app.")

    st.markdown("### Feature Fusion and Classification")
    st.markdown(
        """
        <div class="article-note">
        The contextual representations obtained from the three graphs are combined to form a
        unified multimodal representation. This fused feature vector is then passed through a
        classification layer to produce the final prediction. During inference, new samples are
        connected to the graph using kNN relationships, allowing the model to utilize
        dataset-level contextual information for classification.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Results")
    st.markdown(
        f"""
        <div class="article-note">
        Preliminary experiments were conducted to evaluate the effect of incorporating
        class-semantic relationships before implementing the full multi-graph architecture.
        At this stage, the modality-specific graphs were not included. Three model variations
        were evaluated: PCA Only, PCA + LDA, and PCA + LDA + Graph. The results indicate that
        incorporating class-semantic neighborhood information improves the discriminative
        capability of the model compared to using feature representations alone. In the current
        saved runs in this repository, the full multi-graph model reaches a validation macro F1
        of <strong>{graph_row['val_f1_macro']:.4f}</strong>, while the strongest available score is
        <strong>{best_row['val_f1_macro']:.4f}</strong> from <strong>{best_row['model']}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Code snippet: final prediction")
    st.code(
        """combined_features = torch.cat(
    [image_graph_features, text_graph_features, lda_graph_features], dim=1
)
logits = self.classification_layer(combined_features)""",
        language="python",
    )

    st.markdown("#### Code snippet: app inference")
    st.code(
        """predictor = GraphProjectPredictor('text_image_graph', device='cpu')
result = predictor.predict(caption, image)
print(result['predicted_label'])""",
        language="python",
    )

    st.markdown("### Reference links")
    st.markdown(
        """
        - Literature Review Summary: kNN-KGE (Wang et al., 2023)
        - Learning on Multimodal Graphs: A Survey
        - KNN-GNN: A powerful graph neural network enhanced by aggregating K-nearest neighbors in common subspace
        - k-Nearest Neighbor Learning with Graph Neural Networks
        - https://arxiv.org/pdf/2508.03732
        - GitHub Repository: https://github.com/vishnurchityala/graph-networks/tree/gnn
        """
    )


def render_experiments(selected_model_key: str, experiment_df: pd.DataFrame) -> None:
    st.markdown("### Model results")
    st.dataframe(
        experiment_df.drop(columns=["model_key"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Training curve")
    log_df = get_log_df(selected_model_key)
    metric = st.selectbox(
        "Metric",
        ["val_f1_macro", "val_acc", "train_loss", "val_loss", "train_f1_macro"],
        index=0,
    )

    chart_df = log_df[["epoch", metric]].set_index("epoch")
    st.line_chart(chart_df)

    spec = MODEL_SPECS[selected_model_key]
    st.markdown("### Selected model")
    st.write(spec["title"])
    st.write(spec["summary"])
    st.write(f"Parts used: {', '.join(spec['components'])}")


def render_sample_inputs() -> tuple[object | None, str, str | None]:
    label_display = st.selectbox(
        "Sample class",
        ["Kitchen", "Shopping", "Working", "Leadership"],
    )

    sample_df = get_samples_for_label(label_display)
    sample_options = sample_df["sample_name"].tolist()
    selected_name = st.selectbox("Sample", sample_options)
    sample_row = sample_df.loc[sample_df["sample_name"] == selected_name].iloc[0]

    current_token = f"{sample_row['sample_id']}"
    if st.session_state.get("active_sample_token") != current_token:
        st.session_state["active_sample_token"] = current_token
        st.session_state["sample_caption_input"] = sample_row["image_caption"]

    image = load_pil_image(sample_row["image_path"])
    st.image(image, caption=sample_row["image_path"], use_container_width=True)

    caption = st.text_area(
        "Caption",
        key="sample_caption_input",
        height=120,
    )
    ground_truth = sample_row["label_display"]
    return image, caption, ground_truth


def render_upload_inputs() -> tuple[object | None, str, str | None]:
    upload = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp"],
    )
    caption = st.text_area(
        "Caption",
        key="upload_caption_input",
        placeholder="Add the text paired with the uploaded image...",
        height=120,
    )

    image = None
    if upload is not None:
        image = load_pil_image(upload)
        st.image(image, caption=upload.name, use_container_width=True)

    return image, caption, None


def render_prediction(result: dict[str, object], ground_truth: str | None = None) -> None:
    probability_df = (
        pd.DataFrame(
            [
                {"label": label, "probability": probability}
                for label, probability in result["probabilities"].items()
            ]
        )
        .sort_values("probability", ascending=False)
        .set_index("label")
    )

    st.metric("Predicted label", str(result["predicted_label"]))
    st.metric("Confidence", f"{result['confidence']:.2%}")

    if ground_truth:
        if ground_truth == result["predicted_label"]:
            st.success(f"Prediction matches the sample label: {ground_truth}")
        else:
            st.warning(
                f"Ground truth label: {ground_truth}. "
                f"Predicted: {result['predicted_label']}."
            )

    st.bar_chart(probability_df)
    st.dataframe(probability_df.reset_index(), use_container_width=True, hide_index=True)


def render_inference(selected_model_key: str, runtime_device: str) -> None:
    st.markdown("### Try the model")
    st.markdown(
        f'<p class="small-note">Running `{MODEL_SPECS[selected_model_key]["title"]}` on `{runtime_device}`.</p>',
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Input source",
        ["Project sample", "Upload custom input"],
        horizontal=True,
    )

    if input_mode == "Project sample":
        image, caption, ground_truth = render_sample_inputs()
    else:
        image, caption, ground_truth = render_upload_inputs()

    run_prediction = st.button("Run inference", type="primary", use_container_width=True)

    if run_prediction:
        if image is None:
            st.error("Please choose a sample or upload an image.")
            return
        if not caption.strip():
            st.error("Please provide a caption before running inference.")
            return

        with st.spinner("Loading checkpoint and scoring the sample..."):
            predictor = get_predictor(selected_model_key, runtime_device)
            result = predictor.predict(caption, image)
        render_prediction(result, ground_truth=ground_truth)


def main() -> None:
    dataset_df = get_dataset()
    experiment_df = get_experiment_table()
    device_options = ["cuda", "cpu"] if get_device() == "cuda" else ["cpu"]

    with st.sidebar:
        st.title("Graph App")
        section = st.radio(
            "Navigate",
            ["Overview", "Experiments", "Inference"],
        )
        selected_model_key = st.selectbox(
            "Checkpoint",
            list(MODEL_SPECS.keys()),
            format_func=lambda key: MODEL_SPECS[key]["title"],
        )
        runtime_device = st.selectbox(
            "Runtime device",
            device_options,
            index=0,
        )
        st.caption(MODEL_SPECS[selected_model_key]["summary"])

    if section == "Overview":
        render_overview(dataset_df, experiment_df)
    elif section == "Experiments":
        render_experiments(selected_model_key, experiment_df)
    else:
        render_inference(selected_model_key, runtime_device)


if __name__ == "__main__":
    main()
