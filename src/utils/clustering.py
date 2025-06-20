from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import HDBSCAN, KMeans
from utils import config


def apply(texts: list[str], activations: list[float] | None = None) -> list[list[dict]]:
    """Apply clustering to a list of texts using the specified method.

    This function encodes the input texts into embeddings and applies clustering
    to group similar texts together.

    Args:
        texts: A list of strings to be clustered.
        embedder: A SentenceTransformer object used to encode the texts into embeddings.
        method: The clustering algorithm to use. Options are:
            - "KMeans": Uses KMeans clustering
            - "HDBSCAN": Uses HDBSCAN clustering

    Returns:
        clustered_sentences: A list of clusters, where each cluster is a list of strings from the
          original texts list that belong to the same cluster.

    Raises:
        ValueError: If an unknown clustering method is specified.

    Note:
        Adapted from:
        https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/kmeans.py
    """
    method = config.CLUSTER_METHOD
    if config.CLUSTER_EMBEDDING_MODEL_NAME == None:
        corpus_embeddings = activations.cpu()
    else:
        embedder = SentenceTransformer(
            config.CLUSTER_EMBEDDING_MODEL_NAME, trust_remote_code=True
        )
        embedder.max_seq_length = config.CLUSTER_MAX_SEQ_LEN
        corpus_embeddings = embedder.encode(texts, batch_size=config.BATCH_SIZE)

    if method == "KMeans":
        clustering_model = KMeans(n_clusters=config.N_CLUSTERS)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        n_cluster = config.N_CLUSTERS
    elif method == "HDBSCAN":
        clustering_model = HDBSCAN(min_cluster_size=2)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        n_cluster = len(set(cluster_assignment)) - (
            1 if -1 in cluster_assignment else 0
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    clustered_sentences: list[list[dict]] = [[] for _ in range(n_cluster)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(
            {
                "sentence_id": sentence_id,
                "text": texts[sentence_id],
            }
        )

    return clustered_sentences


def get_cosine_similarity(descriptions, embedder):
    """Compute the cosine similarity matrix and average similarity score for a list of n descriptions.

    Args:
        descriptions (list): List of n strings.

    Returns:
        tuple: (similarity_matrix (torch.Tensor), average_similarity (float))
    """
    corpus_embeddings = embedder.encode(
        descriptions, batch_size=config.BATCH_SIZE, convert_to_tensor=True
    )

    # Compute pairwise cosine similarity matrix
    similarity_matrix = util.pytorch_cos_sim(corpus_embeddings, corpus_embeddings)

    # Compute average similarity excluding the diagonal
    n = similarity_matrix.shape[0]
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += similarity_matrix[i, j].item()
            count += 1
    average_similarity = total / count

    return similarity_matrix, average_similarity
