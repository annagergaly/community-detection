import math

import numpy
import networkx
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from prettytable import PrettyTable
import matplotlib.pyplot as plt


def generate_data_bernoulli(n, t, graph, mu, nu, labels):
    data = numpy.zeros((n, n, t), int)
    for (i, j) in graph.edges:
        if labels[i] == labels[j]:
            data[i, j] = numpy.random.binomial(1, mu, t)
        else:
            data[i, j] = numpy.random.binomial(1, nu, t)
        data[j, i] = data[i, j]
    return data


def generate_data_markov(n, t, graph, mu, nu, trans_in, trans_out, labels):
    data = numpy.zeros((n, n, t), int)
    for (i, j) in graph.edges:
        if labels[i] == labels[j]:
            data[i, j, 0] = numpy.random.binomial(1, mu)
            for k in range(1, t):
                data[i, j, k] = numpy.random.binomial(1, trans_in[data[i, j, k-1], 1])
        else:
            data[i, j, 0] = numpy.random.binomial(1, nu)
            for k in range(1, t):
                data[i, j, k] = numpy.random.binomial(1, trans_out[data[i, j, k - 1], 1])
        data[j, i] = data[i, j]
    return data


def kl_bernoulli(p, q):
    return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))


def js_bernoulli(p, q):
    m = (p+q)/2
    return kl_bernoulli(p, m) + kl_bernoulli(q, m)


def js_markov(p_01, p_11, pi_1, pi_0, i, j, k):
    return (pi_0[i, k] * kl_bernoulli(p_01[i, k], (p_01[i, k]+p_01[k, j]) / 2) +
            pi_1[i, k] * kl_bernoulli(p_11[i, k], (p_11[i, k]+p_11[k, j]) / 2) +
            pi_0[k, j] * kl_bernoulli(p_01[k, j], (p_01[k, j] + p_01[i, k]) / 2) +
            pi_1[k, j] * kl_bernoulli(p_11[k, j], (p_11[i, k] + p_11[k, j]) / 2))


def compute_divergences_bernoulli(a, n, t):
    interactions = numpy.sum(a, axis=2)
    divergences = numpy.ones((n, n)) * -1
    for i in range(n):
        for j in range(i + 1, n):
            size = 0
            divergence = 0
            for k in range(n):
                if interactions[i, k] != 0 and interactions[k, j] != 0:
                    size += 1
                    divergence += js_bernoulli(interactions[i, k] / t, interactions[j, k] / t)
            if size != 0:
                divergences[i, j] = divergences[j, i] = divergence / size
    return divergences


def compute_divergences_markov(a, n, t):
    num_1 = numpy.sum(a, axis=2)
    num_0 = numpy.ones((n, n))*t - num_1
    num_01 = numpy.sum(numpy.diff(a) == 1, axis=2)
    num_11 = numpy.sum(numpy.logical_and(numpy.diff(a, append=numpy.zeros((n, n, 1))) == 0, a == 1), axis=2)
    p_hat_01 = num_01 / num_0
    p_hat_11 = num_11 / num_1
    pi_hat_1 = p_hat_01 / (1 + p_hat_01 - p_hat_11)
    pi_hat_0 = 1 - pi_hat_1
    divergences = numpy.ones((n, n)) * -1
    for i in range(n):
        for j in range(i + 1, n):
            size = 0
            divergence = 0
            for k in range(n):
                if num_1[i, k] != 0 and num_1[k, j] != 0:
                    size += 1
                    divergence += js_markov(p_hat_01, p_hat_11, pi_hat_1, pi_hat_0, i, j, k)
            if size != 0:
                divergences[i, j] = divergences[j, i] = divergence / size
    return divergences


def compute_edge_labels(a, n, t, e=-1.0):
    edge_labels = numpy.zeros((n, n))
    divergences = compute_divergences_bernoulli(a, n, t)
    if e == -1:
        values = divergences[divergences != -1]
        kmeans = KMeans(2).fit(values.reshape(-1, 1))
        cluster_id = numpy.argmin(kmeans.cluster_centers_)
        edge_labels[numpy.isin(divergences, values[kmeans.labels_ == cluster_id])] = 1
    else:
        edge_labels[numpy.logical_and(divergences <= e, divergences != -1)] = 1
    return edge_labels


def calculate_w(div, method="expvar2"):
    match method:
        case "exp":
            w = numpy.exp(-div)
            w[div == -1] = 0
        case "expvar":
            var = numpy.var(div != -1)
            w = numpy.exp(-div / math.sqrt(var))
            w[div == -1] = 0
        case "expvar2":
            var = numpy.var(div != -1)
            w = numpy.exp(-numpy.square(div) / var)
            w[div == -1] = 0
        case _:
            w = div
    return w


def spectral_cluster_from_divergences(w, c):
    return SpectralClustering(n_clusters=c, affinity='precomputed', assign_labels='cluster_qr').fit_predict(w)


def spectral_cluster_combined(a, t, w, weights, c):
    w *= weights[2]
    w += weights[0] * numpy.sum(a, axis=2) * (1/t)
    w += weights[1] * numpy.sum(a[numpy.diff(a, append=numpy.zeros((n, n, 1))) == 0]) * (1/t)
    return SpectralClustering(n_clusters=c, affinity='precomputed', assign_labels='cluster_qr').fit_predict(w)


def spectral_cluster_aggregate(a, c):
    interactions = numpy.sum(a, axis=2)
    return SpectralClustering(n_clusters=c, affinity='precomputed', assign_labels='cluster_qr').fit_predict(interactions)


def gather_neighbourhood(vertex, edge_labels):
    return numpy.flatnonzero(edge_labels[vertex])


def delta_good(cluster, vertex, delta, edge_labels):
    c_size = cluster.size
    neighbourhood = gather_neighbourhood(vertex, edge_labels)
    n_size = neighbourhood.size
    in_c = numpy.intersect1d(neighbourhood, cluster)
    in_size = in_c.size
    if in_size >= (1-delta)*c_size and (n_size-in_size) <= delta*c_size:
        return True
    else:
        return False


def all_good(cluster, delta, edge_labels):
    for c in cluster:
        if not delta_good(cluster, c, delta, edge_labels):
            return False
    return True


def correlation_clustering_pivot(n, edge_labels):
    nodes = numpy.arange(n)
    clusters = []
    while nodes.size > 0 and edge_labels.sum() > 0:
        v = numpy.random.choice(nodes)
        cluster = gather_neighbourhood(v, edge_labels)
        nodes = numpy.setdiff1d(nodes, cluster)
        clusters.append(cluster)
        for c in cluster:
            edge_labels[c] = numpy.zeros(n)
            edge_labels[:, c] = numpy.zeros(n)
    return clusters_to_labels(n, clusters)


def correlation_clustering_cautious(n, delta, edge_labels):
    clusters = []
    available = numpy.arange(n)
    could_not_form_cluster = []
    while available.size - len(could_not_form_cluster) > 0:
        v = numpy.random.choice(available)
        cluster = gather_neighbourhood(v, edge_labels)

        # vertex removal step - could be optimized
        while not all_good(cluster, 3*delta, edge_labels):
            for c in cluster:
                if not delta_good(cluster, c, 3*delta, edge_labels):
                    cluster = numpy.delete(cluster, cluster == c)

        # vertex addition step
        y = []
        for i in available:
            if delta_good(cluster, i, 7 * delta, edge_labels):
                y.append(i)
        cluster = numpy.union1d(cluster, y)

        # delete the formed cluster
        if cluster.size == 0:
            could_not_form_cluster.append(v)
        else:
            available = numpy.setdiff1d(available, cluster)
            could_not_form_cluster.clear()
            clusters.append(cluster)
            for c in cluster:
                edge_labels[c] = numpy.zeros(n)
                edge_labels[:, c] = numpy.zeros(n)
    return clusters_to_labels(n, clusters)


def clusters_to_labels(n, clusters):
    labels = numpy.zeros(n, int)
    for i in range(len(clusters)):
        for c in clusters[i]:
            labels[c] = i+1
    return labels


def labels_to_clusters(c, labels):
    clusters = [[] for _ in range(c)]
    for i in range(len(labels)):
        clusters[labels[i]-1].append(i)
    return clusters


def evaluate_clusterings(data, n, t, method):
    results = numpy.zeros(3)
    # divergences_bernoulli = compute_divergences_bernoulli(data, n, t)
    divergences_markov = compute_divergences_markov(data, n, t)
    # edges = compute_edge_labels(data, n, t)
    # cautious = correlation_clustering_cautious(n, 1 / 3.5, edges)
    # pivot = correlation_clustering_pivot(n, edges)
    # w_b = calculate_w(divergences_bernoulli, method)
    w_m = calculate_w(divergences_markov, method)
    # bernoulli = spectral_cluster_from_divergences(w_b, c)
    markov = spectral_cluster_from_divergences(w_m, c)
    # bernoulli_combined = spectral_cluster_combined(data, t, w_b, weights, c)
    markov_combined = spectral_cluster_combined(data, t, w_m, weights, c)
    spectral_aggregated = spectral_cluster_aggregate(data, c)
    # results[0] = metrics.adjusted_mutual_info_score(labels, bernoulli)
    results[0] = metrics.adjusted_mutual_info_score(labels, markov)
    results[1] = metrics.adjusted_mutual_info_score(labels, spectral_aggregated)
    # results[3] = metrics.adjusted_mutual_info_score(labels, bernoulli_combined)
    results[2] = metrics.adjusted_mutual_info_score(labels, markov_combined)
    return results
    # results['correlation_cautious'][i] = metrics.adjusted_mutual_info_score(label, cautious)
    # results['correlation_pivot'][i] = metrics.adjusted_mutual_info_score(label, pivot)


if __name__ == '__main__':
    dat_dtype = {
        'names': ('a', 'b', 'p_in01', 'p_in11', 'p_out01', 'p_out11', 'method', 'aggregated',  'markov', 'markov_c'),
        'formats': ('i', 'i', 'd', 'd', 'd', 'd', '|U12', 'd', 'd', 'd')}
    methods = ["expvar"]
    change = 1
    c = 2
    n = 100
    t = 500
    a = 6
    b = 6
    transition_in = [numpy.array([[0.8, 0.2],
                                 [0.2, 0.8]]) for _ in range(9)
                     ]
    transition_out = [numpy.array([[1-i*0.1, i*0.1],
                                  [i*0.1, 1-i*0.1]]) for i in range(1, 10)
                      ]
    mu = 0.4
    nu = 0.4
    weights = [1, 1, 1]
    repeat = 20
    results = numpy.zeros(change*len(methods)*len(transition_in), dat_dtype)

    for i in range(change):
        p = (math.log(n) / n) * a
        q = (math.log(n) / n) * b
        scores = numpy.zeros(3)
        for j in range(len(transition_in)):
            for k in range(len(methods)):
                for _ in range(repeat):
                    labels = numpy.random.choice(c, n)
                    clusters = labels_to_clusters(c, labels)
                    interaction_graph = networkx.stochastic_block_model([len(i) for i in clusters], numpy.eye(c) * (p - q) + q,
                                                                        [item for sublist in clusters for item in sublist])
                    data = generate_data_markov(n, t, interaction_graph, mu, nu, transition_in[j], transition_out[j], labels)
                    scores += evaluate_clusterings(data, n, t, methods[k])
                # results['bernoulli'][i*len(transition_in)*len(methods)+j*len(methods)+k] = scores[0]/repeat
                results['markov'][i*len(transition_in)*len(methods)+j*len(methods)+k] = scores[0]/repeat
                results['aggregated'][i*len(transition_in)*len(methods)+j*len(methods)+k] = scores[1]/repeat
                # results['bernoulli_c'][i*len(transition_in)*len(methods)+j*len(methods)+k] = scores[3] / repeat
                results['markov_c'][i*len(transition_in)*len(methods)+j*len(methods)+k] = scores[2] / repeat
                results['a'][i*len(transition_in)*len(methods)+j*len(methods)+k] = a
                results['b'][i*len(transition_in)*len(methods)+j*len(methods)+k] = b
                results['p_in01'][i*len(transition_in)*len(methods)+j*len(methods)+k] = transition_in[j][0, 1]
                results['p_in11'][i*len(transition_in)*len(methods)+j*len(methods)+k] = transition_in[j][1, 1]
                results['p_out01'][i*len(transition_in)*len(methods)+j*len(methods)+k] = transition_out[j][0, 1]
                results['p_out11'][i*len(transition_in)*len(methods)+j*len(methods)+k] = transition_out[j][1, 1]
                results['method'][i*len(transition_in)*len(methods)+j*len(methods)+k] = methods[k]
        b += 1
        a += 1

    x = PrettyTable(results.dtype.names)
    for row in results:
        x.add_row(row)
    # x.float_format = ".4"
    print(x)

    plt.plot([i*0.1 for i in range(1, 10)], results['markov'], 'r-',
             [i*0.1 for i in range(1, 10)], results['markov_c'], 'b-',
             [i*0.1 for i in range(1, 10)], results['aggregated'], 'k-')
    plt.show()
