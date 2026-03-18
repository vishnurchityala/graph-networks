# Novelty of the Proposed Multi-Graph Contextual Classification Architecture

## Overview

Traditional multimodal classification models typically treat each sample independently by extracting embeddings from text and images and applying a fusion network for prediction. While these approaches capture semantic features, they do not explicitly utilize the **statistical relationships between samples within the dataset**.

The proposed architecture introduces a **multi-graph contextual learning framework** that models relational structure in the dataset through three complementary graphs:

1. **Text Semantic Graph**
2. **Visual Semantic Graph**
3. **Class-Discriminative Graph**

Each graph captures a different statistical structure of the dataset and is processed through a **Graph Attention Network (GAT)** to enable context-aware feature propagation. The outputs from these graphs are then combined for final classification.

This design transforms the classification task from **independent feature prediction** into **context-aware relational inference over multimodal statistical neighborhoods**.

---

# Multi-Graph Representation of Dataset Structure

The central idea of the architecture is to represent the dataset through **multiple relational manifolds**, where each graph captures a distinct form of similarity among samples.

## Text Semantic Graph

The text semantic graph models relationships between samples based on linguistic similarity.

Text captions are encoded using BERT embeddings and then reduced using PCA to obtain compact text representations. A K-nearest neighbor graph is constructed using cosine similarity between these vectors.

This graph captures:

* similarity of misogynistic language patterns
* recurring textual stereotypes
* linguistic contextual relationships between captions

Applying graph attention allows the model to learn how information should propagate among semantically related captions.

---

## Visual Semantic Graph

The visual semantic graph represents similarity relationships between images.

Images are encoded using CLIP visual embeddings and reduced using PCA to produce compact visual feature vectors. A KNN graph is then constructed using cosine similarity.

This graph captures visual relationships such as:

* stereotypical scenes
* recurring visual patterns associated with misogyny
* contextual similarities between meme images

Graph attention allows the model to adaptively weight visual neighbors, enabling contextual refinement of image representations.

---

## Class-Discriminative Graph

The third graph represents **class-level statistical relationships** between samples.

Unlike the previous two graphs that capture semantic similarity, this graph focuses on **discriminative class structure**.

The process is as follows:

1. Text and image PCA features are combined into a multimodal representation.
2. Linear Discriminant Analysis (LDA) is applied to project the data into a space that maximizes class separability.
3. A K-nearest neighbor graph is constructed using cosine similarity in this LDA space.

The LDA projection optimizes the ratio between between-class and within-class variance:

```text
max_W |W^T S_B W| / |W^T S_W W|
```

Equivalent mathematical form: `max_W |W^T S_B W| / |W^T S_W W|`.

Where:

* $S_B$ represents between-class scatter
* $S_W$ represents within-class scatter
* $W$ is the projection matrix

This transformation ensures that samples belonging to the same class cluster together while samples from different classes separate.

The resulting graph therefore encodes **class proximity relationships** rather than raw semantic similarity.

---

# Contextual Feature Propagation using Graph Attention

After graph construction, each graph is processed using a Graph Attention Network.

Graph attention allows nodes to aggregate information from neighboring samples while learning **adaptive attention weights** that determine the importance of each neighbor.

The feature propagation step is defined as:

```text
h_i' = sum_{j in N(i)} alpha_ij W h_j
```

Equivalent mathematical form: `h_i' = sum_{j in N(i)} alpha_ij W h_j`.

where:

* $h_i$ is the feature of node $i$
* $W$ is a learned transformation
* $N(i)$ represents neighboring nodes
* $\alpha_{ij}$ is the learned attention weight

Unlike traditional KNN methods, where neighbors influence predictions through static voting or distance weighting, graph attention enables **learned relational reasoning** over the dataset.

This allows the model to selectively focus on the most informative neighbors while ignoring noisy or misleading samples.

---

# Difference from Traditional KNN Approaches

Although the graph structure is initially constructed using a KNN procedure, the resulting model differs fundamentally from classical KNN classification.

Traditional KNN performs prediction by directly voting over the labels of neighboring samples. This method does not involve feature learning and treats all neighbors with fixed importance.

In contrast, the proposed architecture uses KNN only to construct the graph topology. Feature propagation through Graph Attention Networks allows the model to:

* learn adaptive weighting over neighbors
* aggregate neighbor features rather than labels
* refine representations through trainable transformations
* produce non-linear classification boundaries

As a result, the system learns **contextual representations influenced by the local statistical structure of the dataset** rather than relying on static nearest-neighbor voting.

---

# Multi-View Relational Reasoning

The three-graph architecture enables **multi-view relational reasoning**.

Each graph captures a different aspect of the dataset:

| Graph Type  | Captured Relationship          |
| ----------- | ------------------------------ |
| Text Graph  | linguistic semantic similarity |
| Image Graph | visual semantic similarity     |
| LDA Graph   | class discriminative proximity |

By processing these graphs independently, the model preserves modality-specific relational information before combining them for final classification.

This design allows the classifier to simultaneously reason over:

* textual context
* visual context
* class structure

Such multi-view relational modeling is particularly beneficial for multimodal meme analysis where textual and visual cues often convey complementary or conflicting signals.

---

# Hybrid Statistical–Deep Learning Pipeline

Another distinctive characteristic of the proposed architecture is the integration of **classical statistical methods with deep neural models**.

The pipeline combines:

* pretrained multimodal encoders (BERT and CLIP)
* statistical dimensionality reduction (PCA)
* discriminative statistical projection (LDA)
* graph-based contextual learning (GAT)

This hybrid approach leverages the strengths of both statistical learning and deep neural representation learning.

Statistical transformations improve feature separability and reduce noise before graph construction, resulting in more reliable neighborhood structures for graph learning.

---

# Dataset-Level Contextual Memory

The proposed architecture implicitly transforms the training dataset into a **relational memory structure**.

Each sample is represented as a node in the graph and connected to statistically similar samples.

During inference, new samples are connected to the training graph through nearest-neighbor relationships, allowing the model to incorporate contextual information from existing examples.

This enables predictions that depend not only on the individual sample but also on the **local statistical context of the dataset**.

---

# Summary of Contributions

The proposed model introduces several novel aspects:

1. A **multi-graph multimodal architecture** that models textual, visual, and class-discriminative relationships separately.

2. A **class-aware graph construction strategy** using LDA projections to encode discriminative statistical relationships between samples.

3. A **context-aware classification mechanism** based on Graph Attention Networks that learns adaptive influence from neighboring samples.

4. A **hybrid statistical-deep learning pipeline** combining PCA, LDA, and graph neural networks for robust multimodal representation learning.

5. A **dataset-level relational reasoning framework** where classification decisions are influenced by the statistical neighborhood structure of the dataset.
