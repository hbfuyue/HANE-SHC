import numpy as np
import math
from heapq import nlargest
from heapq import nsmallest

##################################################
#函数load_data_toydata(neighbours_file, neighbours_NX_file, edges_file)，
##################################################
def load_data_toydata(neighbours_file, neighbours_NX_file, edges_file):
    # 定义节点列表
    node_list = []
    maxNeighbors = 0
    minNeighbors = np.inf
    with open(neighbours_NX_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            node_list.append(node_adj_line[0])
            lenNeighbors = len(node_adj_line)
            if maxNeighbors < lenNeighbors:
                maxNeighbors = lenNeighbors
            if minNeighbors > lenNeighbors:
                minNeighbors = lenNeighbors

    node_list_len = len(node_list)
    # 定义伪节点列表
    node_X = []
    for i in range(0, maxNeighbors - minNeighbors):
        node_X.append('X'+str(i))
    node_X_len = len(node_X)
    #
    node_list_X = node_list + node_X

    # node_vector定
    node_vector = np.zeros((node_list_len, node_list_len)).tolist()
    #
    position_node_i = []
    for i in range(node_list_len):
        x = [i]
        position_node_i.append(x)

    for i in range(node_list_len):
        for j in range(len(position_node_i[i])):
            node_vector[i][position_node_i[i][j]] = node_vector[i][position_node_i[i][j]] + 1.0
    print('node_vector')
    print(node_vector)
    X_zero = np.zeros((node_X_len, node_list_len)).tolist()
    for i in range(node_X_len):
        node_vector.append(X_zero[i])

    node_neighbours = []  #
    node_neighbours_vectors = []  #
    node_neighbours_NX = []  #
    node_neighbours_vectors_NX = []  #
    A = []
    D = []  # D
    DAD = []
    # node_neighbours_ture_numbers = []
    with open(neighbours_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            H_neigher_node_vector = []  #
            node_neighbours_i = []  #
            for i in range(len(node_adj_line)):
                node_neighbours_i.append(node_adj_line[i])
                node_position = node_list_X.index(node_adj_line[i])
                H_node_vector_i = node_vector[node_position]
                H_neigher_node_vector.append(H_node_vector_i)
            node_neighbours.append(node_neighbours_i)
            node_neighbours_vectors.append(H_neigher_node_vector)

    with open(neighbours_NX_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            H_neigher_node_vector = []  #
            node_neighbours_i = []  #
            for i in range(len(node_adj_line)):
                node_neighbours_i.append(node_adj_line[i])
                node_position = node_list.index(node_adj_line[i])
                H_node_vector_i = node_vector[node_position]
                H_neigher_node_vector.append(H_node_vector_i)
            node_neighbours_NX.append(node_neighbours_i)
            node_neighbours_vectors_NX.append(H_neigher_node_vector)

    node_negtive_nodes = []
    node_negtive_nodes_vectors = []
    node_list_set = set(node_list)
    for i in range(len(node_list) - 1):
        node_negtive_nodes_i = list(node_list_set - set(node_neighbours_NX[i]))
        node_negtive_nodes.append(node_negtive_nodes_i)
        node_negtive_nodes_vectors_i = []
        for j in range(len(node_negtive_nodes_i)):
            position_i = node_list.index(node_negtive_nodes_i[j])
            node_negtive_nodes_vectors_i.append(node_vector[position_i])
        node_negtive_nodes_vectors.append(node_negtive_nodes_vectors_i)

    # print(len(node_neighbours_vectors))

    #
    for i in range(node_list_len):
        a = np.zeros((maxNeighbors, maxNeighbors)).tolist()
        A.append(a)

    NODE_neighbours = []
    with open(neighbours_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            NODE_neighbours.append(node_adj_line)

    with open(edges_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            a_index = node_list.index(node_adj_line[0])
            a_i = NODE_neighbours[a_index].index(node_adj_line[1])
            a_j = NODE_neighbours[a_index].index(node_adj_line[2])
            A[a_index][a_i][a_j] = 1.0
            A[a_index][a_j][a_i] = 1.0

    # print("---NODE_neighbours---")
    # print(NODE_neighbours)
    print("---A---")
    print(A)
    # 计算D
    for i in range(node_list_len):
        d_i = np.array(np.sum(A[i], axis=0))
        for i in range(maxNeighbors):
            if d_i[i] == 0:
                pass
            else:
                d_i[i] = d_i[i] ** -1 / 2
        d_hat = np.diag(d_i)
        D.append(d_hat)
    print("---D---")
    print(D)

    # 计算DAD

    for i in range(node_list_len):
        dad = D[i] * A[i] * D[i]
        # dad = D[i] ** -1/2 *
        DAD.append(dad)
    print("---DAD---")
    print(DAD)
    return DAD, node_list, node_list_X, node_vector, node_neighbours, node_neighbours_vectors, node_neighbours_NX, node_neighbours_vectors_NX, node_negtive_nodes, node_negtive_nodes_vectors

##################################################
#函数load_data(neighbours_file, neighbours_NX_file, edges_file)，
##################################################
def load_data(neighbours_file, neighbours_NX_file, edges_file):
    # 定义节点列表
    node_list = []
    maxNeighbors = 0
    minNeighbors = np.inf
    with open(neighbours_NX_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            node_list.append(node_adj_line[0])
            lenNeighbors = len(node_adj_line) - 1
            if maxNeighbors < lenNeighbors:
                maxNeighbors = lenNeighbors
            if minNeighbors > lenNeighbors:
                minNeighbors = lenNeighbors

    node_list_len = len(node_list)
    # 定义伪节点列表
    node_X = []
    for i in range(0, maxNeighbors - minNeighbors):
        node_X.append('X'+str(i))
    node_X_len = len(node_X)
    #
    node_list_X = node_list + node_X


    #
    node_vector = np.zeros((node_list_len, node_list_len)).tolist()
    #
    position_node_i = []
    for i in range(node_list_len):
        x = [i]
        position_node_i.append(x)

    for i in range(node_list_len):
        for j in range(len(position_node_i[i])):
            node_vector[i][position_node_i[i][j]] = node_vector[i][position_node_i[i][j]] + 1.0
    print('node_vector')
    print(node_vector)
    X_zero = np.zeros((node_X_len, node_list_len)).tolist()
    for i in range(node_X_len):
        node_vector.append(X_zero[i])

    node_neighbours = []  #
    node_neighbours_vectors = []  #
    node_neighbours_NX = []  #
    node_neighbours_vectors_NX = []  #
    A = []
    D = []  #
    DAD = []
    # node_neighbours_ture_numbers = []
    with open(neighbours_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            H_neigher_node_vector = []  #
            node_neighbours_i = []  #
            for i in range(len(node_adj_line) - 1):
                node_neighbours_i.append(node_adj_line[i])
                node_position = node_list_X.index(node_adj_line[i])
                H_node_vector_i = node_vector[node_position]
                H_neigher_node_vector.append(H_node_vector_i)
            node_neighbours.append(node_neighbours_i)
            node_neighbours_vectors.append(H_neigher_node_vector)

    with open(neighbours_NX_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            H_neigher_node_vector = []  #
            node_neighbours_i = []  #
            for i in range(len(node_adj_line) - 1):
                node_neighbours_i.append(node_adj_line[i])
                node_position = node_list.index(node_adj_line[i])
                H_node_vector_i = node_vector[node_position]
                H_neigher_node_vector.append(H_node_vector_i)
            node_neighbours_NX.append(node_neighbours_i)
            node_neighbours_vectors_NX.append(H_neigher_node_vector)

    node_negtive_nodes = []
    node_negtive_nodes_vectors = []
    node_list_set = set(node_list)
    for i in range(len(node_list)):
        node_negtive_nodes_i = list(node_list_set - set(node_neighbours_NX[i]))
        node_negtive_nodes.append(node_negtive_nodes_i)
        node_negtive_nodes_vectors_i = []
        for j in range(len(node_negtive_nodes_i)):
            position_i = node_list.index(node_negtive_nodes_i[j])
            node_negtive_nodes_vectors_i.append(node_vector[position_i])
        node_negtive_nodes_vectors.append(node_negtive_nodes_vectors_i)

    # print(len(node_neighbours_vectors))

    #
    for i in range(node_list_len):
        a = np.zeros((maxNeighbors, maxNeighbors)).tolist()
        A.append(a)

    NODE_neighbours = []
    with open(neighbours_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            NODE_neighbours.append(node_adj_line)

    with open(edges_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            a_index = node_list.index(node_adj_line[0])
            a_i = NODE_neighbours[a_index].index(node_adj_line[1])
            a_j = NODE_neighbours[a_index].index(node_adj_line[2])
            A[a_index][a_i][a_j] = 1.0
            A[a_index][a_j][a_i] = 1.0

    # print("---NODE_neighbours---")
    # print(NODE_neighbours)
    print("---A---")
    print(A)
    # 计算D
    for i in range(node_list_len):
        d_i = np.array(np.sum(A[i], axis=0))
        for i in range(maxNeighbors):
            if d_i[i] == 0:
                pass
            else:
                d_i[i] = d_i[i] ** -1 / 2
        d_hat = np.diag(d_i)
        D.append(d_hat)
    print("---D---")
    print(D)

    # 计算DAD

    for i in range(node_list_len):
        dad = D[i] * A[i] * D[i]
        # dad = D[i] ** -1/2 *
        DAD.append(dad)
    print("---DAD---")
    print(DAD)
    return DAD, node_list, node_list_X, node_vector, node_neighbours, node_neighbours_vectors, node_neighbours_NX, node_neighbours_vectors_NX, node_negtive_nodes, node_negtive_nodes_vectors

##################################################
#
##################################################
def relu(x):
    return(abs(x)+x)/2

##################################################
#
##################################################
def get_label_file(label_file):
    labels = []
    labels_kind = set()
    with open(label_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            labels.append(node_adj_line[1])
            labels_kind.add(node_adj_line[1])

    return labels,labels_kind

##################################################
#
##################################################
def get_label_file_int(label_file):
    labels = []
    labels_kind = set()
    with open(label_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            labels.append(int(node_adj_line[1]))
            labels_kind.add(int(node_adj_line[1]))

    return labels,labels_kind

##################################################
#
##################################################
def get_cs_labels(content_label_file,structure_label_file):
    cs_labels = []
    c_labels = []
    with open(content_label_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            c_labels.append(node_adj_line[1])
    f.close()
    s_labels = []
    with open(structure_label_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            s_labels.append(node_adj_line[1])
    f.close()
    for i in range(len(c_labels)):
        cs_labels.append([c_labels[i],s_labels[i]])
    return cs_labels

##################################################
#
##################################################
def read_embedding(embedding_file):
    embedding = []
    with open(embedding_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            embedding.append([float(node_adj_line[1]), float(node_adj_line[2])])
    return embedding

##################################################
#
##################################################
def my_hamming_loss_to_classification(true_label, pred_lable):
    M = len(true_label)
    L = len(true_label[0])
    sum = 0
    for i in range(M):
        for j in range(L):
            if true_label[i][j]!=pred_lable[i][j]:
                sum = sum+1
    hamming_distance = sum/(M*L)
    return hamming_distance

#######################################################
#
#######################################################
def load_data_toyData_H(nodes_file,edges_file, node_positives_file, node_negatives_file):
    #1.
    nodes_list = []
    with open(nodes_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line_split = line.split(',')
            nodes_list.append(line_split[0])
    node_list_len = len(nodes_list)
    #2.
    A = np.zeros((node_list_len, node_list_len))
    #3.
    with open(edges_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line_split = line.split(',')
            position_x = nodes_list.index(line_split[0])
            position_y = nodes_list.index(line_split[1])
            print('position_x,position_y',position_x,position_y)
            A[position_x][position_y] = 1.0
            A[position_y][position_x] = 1.0
    for i in range(node_list_len):
        A[i][i] = 1.0

    #4.
    D = np.zeros((node_list_len, node_list_len))
    for i in range(len(nodes_list)):
        d_i = np.array(np.sum(A, axis=0))
        for i in range(len(nodes_list)):
            if d_i[i] == 0:
                pass
            else:
                d_i[i] = d_i[i] ** -1 / 2
        D = np.diag(d_i)

    #5.
    DAD = D * A * D

    #6.
    H = np.identity(node_list_len).tolist()

    #7.
    nodes_positives = []
    nodes_positive_number = []
    label_nodes_positive = []
    with open(node_positives_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line_split = line.split(',')
            nodes_positives.append(line_split[1:]) #
            nodes_positive_number.append(len(line_split[1:]))
    for i in range(len(nodes_positives)):
        print('A=', i + 1)
        for j in range(len(nodes_positives[i])):
            print(nodes_positives[i][j])
            label_nodes_positive.append(H[nodes_list.index(nodes_positives[i][j])])

    # 8.
    nodes_negatives = []
    nodes_negative_number = []
    label_nodes_negative = []
    with open(node_negatives_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line_split = line.split(',')
            nodes_negatives.append(line_split[1:])  #
            nodes_negative_number.append(len(line_split[1:]))
    for i in range(len(nodes_negatives)):
        print('A=',i+1)
        for j in range(len(nodes_negatives[i])):
            print(nodes_negatives[i][j])
            label_nodes_negative.append(H[nodes_list.index(nodes_negatives[i][j])])

    return DAD,H,nodes_positive_number,label_nodes_positive,nodes_negative_number,label_nodes_negative

#######################################################
#
#######################################################
def load_data_Data_experiment(nodes_file,edges_file, node_positives_file, node_negatives_file):
    #1.
    nodes_list = []
    with open(nodes_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line_split = line.split('\t')
            if line_split[0] not in nodes_list:
                nodes_list.append(line_split[0])
    f.close()
    node_list_len = len(nodes_list)
    #2.
    A = np.zeros((node_list_len, node_list_len)).tolist()
    #3.
    with open(edges_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line_split = line.split('\t')
            position_x = nodes_list.index(line_split[0])
            position_y = nodes_list.index(line_split[1])
            #print('position_x,position_y',position_x,position_y)
            A[position_x][position_y] = 1.0
            A[position_y][position_x] = 1.0
    f.close()
    for i in range(node_list_len):
        A[i][i] = 1.0

    #4.
    '''
    D = np.zeros((node_list_len, node_list_len))
    for i in range(len(nodes_list)):
        d_i = np.array(np.sum(A, axis=0))
        for i in range(len(nodes_list)):
            if d_i[i] == 0:
                pass
            else:
                d_i[i] = d_i[i] ** -1 / 2
        D = np.diag(d_i)
    '''
    D_hat = np.array(np.sum(A, axis=0))  #
    D_hat = np.matrix(np.diag(D_hat))
    D = D_hat ** -1/2

    #5.
    DAD = D * A * D

    #6.
    #H = np.identity(node_list_len).tolist()
    node_zeros = np.zeros((len(nodes_list), len(nodes_list))).tolist()
    for i in range(len(nodes_list)):
        node_zeros[i][i] = node_zeros[i][i] + 1
    H = node_zeros

    #7.
    nodes_positives = []
    nodes_positive_number = []
    label_nodes_positive = []
    with open(node_positives_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line_split = line.split(',')
            endl = len(line_split)
            nodes_positives.append(line_split[1:endl-2]) #
            nodes_positive_number.append(len(line_split[1:endl-2]))
    for i in range(len(nodes_positives)):
        #print('A=', i + 1)
        for j in range(len(nodes_positives[i])):
            #print(nodes_positives[i][j])
            label_nodes_positive.append(H[nodes_list.index(nodes_positives[i][j])])

    # 8.
    nodes_negatives = []
    nodes_negative_number = []
    label_nodes_negative = []
    with open(node_negatives_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line_split = line.split(',')
            endl = len(line_split)
            nodes_negatives.append(line_split[1:endl-2])  #
            nodes_negative_number.append(len(line_split[1:endl-2]))
    for i in range(len(nodes_negatives)):
        #print('A=',i+1)
        for j in range(len(nodes_negatives[i])):
            #print(nodes_negatives[i][j])
            label_nodes_negative.append(H[nodes_list.index(nodes_negatives[i][j])])

    return DAD,H,nodes_positive_number,label_nodes_positive,nodes_negative_number,label_nodes_negative,nodes_list

#
def y_pred_content_modify_func(y_pred_content):
    y_pred_content_list = list(y_pred_content)
    y_pred_content_list_modify = []
    for i in range(len(y_pred_content_list)):
        y_pred_content_list_modify.append('C'+str(y_pred_content_list[i]))
    #print(y_pred_content_list_modify)
    return y_pred_content_list_modify

#
def labels_content_modify_func(labels_content):
    labels_content_list_modify = []
    for i in range(len(labels_content)):
        labels_content_list_modify.append('C' + str(labels_content[i]))
    #print(labels_content_list_modify)
    return labels_content_list_modify

#
def nodeEmbedding_to_edgeEmbedding_by_multiplication(nodeEmbedding, edges_file,nodes_list, outputm):
    #
    edges_list = []
    with open(edges_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split('\t')
            if node_adj_line[1] == node_adj_line[0] or [node_adj_line[1], node_adj_line[0]] in edges_list or [node_adj_line[0], node_adj_line[1]] in edges_list:
                pass
            else:
                edges_list.append([node_adj_line[1], node_adj_line[0]])

    #
    edge_count = len(edges_list)

    noden = len(nodeEmbedding)

    edgeEmbedding = []
    edgeEmbedding_ij = []
    edgeLabel = []
    n = 0
    for i in range(noden):
        for j in range(noden):
            edgeEmbedding_i_j = []
            for k in range(outputm):
                edgeEmbedding_k = nodeEmbedding[i][k] * nodeEmbedding[j][k]
                edgeEmbedding_i_j.append(edgeEmbedding_k)
            if i == j or [nodes_list[i], nodes_list[j]] + edgeEmbedding_i_j in edgeEmbedding_ij or [nodes_list[j], nodes_list[i]] + edgeEmbedding_i_j in edgeEmbedding_ij:
                pass
            else:
                if [nodes_list[i], nodes_list[j]] in edges_list or [nodes_list[j], nodes_list[i]] in edges_list:
                    edgeEmbedding_ij.append([nodes_list[i], nodes_list[j]] + edgeEmbedding_i_j)
                    edgeEmbedding.append(edgeEmbedding_i_j)
                    edgeLabel.append(1)
                else:
                    if n < edge_count:
                        n = n + 1
                        edgeEmbedding_ij.append([nodes_list[i], nodes_list[j]] + edgeEmbedding_i_j)
                        edgeEmbedding.append(edgeEmbedding_i_j)
                        edgeLabel.append(0)

    return edgeEmbedding, edgeLabel, edge_count

#
def nodeEmbedding_to_edgeEmbedding_by_add(nodeEmbedding, edges_file,nodes_list, outputm):
    #
    edges_list = []
    with open(edges_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split('\t')
            if node_adj_line[1] == node_adj_line[0] or [node_adj_line[1], node_adj_line[0]] in edges_list or [
                node_adj_line[0], node_adj_line[1]] in edges_list:
                pass
            else:
                edges_list.append([node_adj_line[1], node_adj_line[0]])

    #
    edge_count = len(edges_list)
    #
    #
    noden = len(nodeEmbedding)
    #
    edgeEmbedding = []
    edgeEmbedding_ij = []
    edgeLabel = []
    n = 0  #
    for i in range(noden):
        for j in range(noden):
            edgeEmbedding_i_j = []
            for k in range(outputm):
                edgeEmbedding_k = (nodeEmbedding[i][k] + nodeEmbedding[j][k]) / 2
                edgeEmbedding_i_j.append(edgeEmbedding_k)
            if i == j or [nodes_list[i], nodes_list[j]] + edgeEmbedding_i_j in edgeEmbedding_ij or [nodes_list[j], nodes_list[i]] + edgeEmbedding_i_j in edgeEmbedding_ij:
                pass
            else:
                if [nodes_list[i], nodes_list[j]] in edges_list or [nodes_list[j], nodes_list[i]] in edges_list:
                    edgeEmbedding_ij.append([nodes_list[i], nodes_list[j]] + edgeEmbedding_i_j)
                    edgeEmbedding.append(edgeEmbedding_i_j)
                    edgeLabel.append(1)
                else:
                    if n < edge_count:
                        n = n + 1
                        edgeEmbedding_ij.append([nodes_list[i], nodes_list[j]] + edgeEmbedding_i_j)
                        edgeEmbedding.append(edgeEmbedding_i_j)
                        edgeLabel.append(0)
    return edgeEmbedding, edgeLabel, edge_count

#
def nodeEmbedding_to_edgeEmbedding_by_connection(nodeEmbedding, edges_file,nodes_list, outputm):

    return True

###############################################################
###################################
###############################################################
#########################
######
#########################

#"
def init_xxx_feature_embedding(n):
    init_xxx_feature_embedding=[]
    for i in range(n):
        init_xxx_feature_embedding.append(0)
    return init_xxx_feature_embedding

#
def distance(a,b):
    n = len(b)
    s = 0
    for i in range(n):
        s = s + (a[i] - b[i])*(a[i] - b[i])
    s = math.sqrt(s)
    return s

#
def get_xxxs_list(node_xxx_feature_file):
    xxxs_list = []
    with open(node_xxx_feature_file, "r", encoding='UTF-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            node_adj_line = line.split('\t')
            xxxs_list.append(node_adj_line[0])
    return xxxs_list

#
def get_xxxs_content_embedding_list(node_xxx_feature_file):
    xxxs_content_embedding_list = []
    xxxs_feature = set()
    with open(node_xxx_feature_file, "r", encoding='UTF-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            node_adj_line = line.split('\t')
            xxx_feature = node_adj_line[4]
            feature_list = xxx_feature.split(";")
            for i in range(len(feature_list)):
                xxxs_feature.add(feature_list[i])
    fr.close()
    xxxs_feature_list = list(xxxs_feature)
    xxxs_feature_list.sort()
    l = len(xxxs_feature)

    with open(node_xxx_feature_file, "r", encoding='UTF-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            node_adj_line = line.split('\t')
            xxx_feature = node_adj_line[4]
            feature_list = xxx_feature.split(";")
            xxx_feature_embedding = init_xxx_feature_embedding(l)
            for i in range(len(feature_list)):
                position = xxxs_feature_list.index(feature_list[i])
                xxx_feature_embedding[position] = 1
            xxxs_content_embedding_list.append(xxx_feature_embedding)
    fr.close()
    return xxxs_content_embedding_list

def get_xxxs_content_embedding_list_to_float(node_xxx_feature_file):
    xxxs_content_embedding_list = []
    xxxs_feature = set()
    with open(node_xxx_feature_file, "r", encoding='UTF-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            node_adj_line = line.split('\t')
            xxx_feature = node_adj_line[4]
            feature_list = xxx_feature.split(";")
            for i in range(len(feature_list)):
                xxxs_feature.add(feature_list[i])
    fr.close()
    xxxs_feature_list = list(xxxs_feature)
    xxxs_feature_list.sort()
    l = len(xxxs_feature)

    with open(node_xxx_feature_file, "r", encoding='UTF-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            node_adj_line = line.split('\t')
            xxx_feature = node_adj_line[4]
            feature_list = xxx_feature.split(";")
            xxx_feature_embedding = init_xxx_feature_embedding(l)
            for i in range(len(feature_list)):
                position = xxxs_feature_list.index(feature_list[i])
                xxx_feature_embedding[position] = 1.0
            xxxs_content_embedding_list.append(xxx_feature_embedding)
    fr.close()
    return xxxs_content_embedding_list
#
def computer_content_simarity_top_k_by_XXXContentEmbedding(k, content_x, content_list, content_object_list):
    ll = len(content_list)
    distance_list = []
    for i in range(ll):
        distance_list.append(distance(content_x, content_list[i]))
    top_k_list=list(set(nsmallest(k, distance_list)))

    idx = []
    for y in top_k_list:
        idx = idx + [i for i, x in enumerate(distance_list) if x == y]

    top_K_content = []
    for y in idx:
        top_K_content.append(content_object_list[y])

    return top_K_content

#
def computer_xxx_content_simarity_top_k_list_to_reprocessfile(top_k, node_xxx_feature_file, xxx_type, based_direction):
    xxx_content_simarity_top_k_list = []

    #0.
    print("0.content_object_list")
    content_object_list = get_xxxs_list(node_xxx_feature_file)

    #1.
    print("1.xxxs_content_embedding_list")
    xxxs_content_embedding_list = get_xxxs_content_embedding_list(node_xxx_feature_file)

    #2.
    print("2.xxx_content_simarity_top_k_list")
    for i in range(len(xxxs_content_embedding_list)):
        print(i)
        xxx_content_simarity_top_k_list.append(computer_content_simarity_top_k_by_XXXContentEmbedding(top_k, xxxs_content_embedding_list[i], xxxs_content_embedding_list, content_object_list))

    print("3.writting to file")
    with open(based_direction+xxx_type+'_content_simarity_top_k_list.txt', 'a') as fw:
        for i in range(len(xxx_content_simarity_top_k_list)):
            ss = ''
            for m in range(len(xxx_content_simarity_top_k_list[i])):
                ss = ss + ',' + xxx_content_simarity_top_k_list[i][m]
            print(str(content_object_list[i]).strip(),ss, file=fw)
    fw.close()

#从
def read_xxx_content_simarity_top_k_list_from_reprocessfile(xxx_content_simarity_top_k_file):
    xxx_content_simarity_top_k_list_reprocess = []
    with open(xxx_content_simarity_top_k_file, 'r',encoding='UTF-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            xxx_content_simarity_top_k_list_reprocess.append(node_adj_line)
    fr.close()
    return xxx_content_simarity_top_k_list_reprocess

#
def get_movies_to_starrings_list(relationship_movies_starrings_file):
    relationship_movies_to_starrings_list = []
    with open(relationship_movies_starrings_file, "r", encoding='UTF-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            relationship_movies_to_starrings_list.append(node_adj_line)
    fr.close()
    return relationship_movies_to_starrings_list

#
def get_movies_to_movies_list(relationship_movies_movies_file):
    relationship_movies_to_movies_list = []
    with open(relationship_movies_movies_file, "r", encoding='UTF-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            relationship_movies_to_movies_list.append(node_adj_line)
    fr.close()
    return relationship_movies_to_movies_list

#
def get_starrings_to_starrings_list(relationship_starrings_starrings_file):
    relationship_starrings_to_starrings_list = []
    with open(relationship_starrings_starrings_file, "r", encoding='UTF-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            relationship_starrings_to_starrings_list.append(node_adj_line)
    fr.close()
    return relationship_starrings_to_starrings_list

#
def get_starrings_to_movies_list(relationship_starrings_movies_file):
    relationship_starrings_to_movies_list = []
    with open(relationship_starrings_movies_file, "r", encoding='UTF-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            relationship_starrings_to_movies_list.append(node_adj_line)
    fr.close()
    return relationship_starrings_to_movies_list







