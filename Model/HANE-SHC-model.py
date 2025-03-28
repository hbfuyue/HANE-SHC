import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import random
from applications_new import *
from utils import *
import time
import matplotlib.pyplot as plt


##############
# 0.
##############

#
based_directory = "../../Data/01_Data_newmovies/movie_starring_1/1/"
nodes_file = based_directory + "He_nodes.txt"
edges_file = based_directory + "He_edges_movie_starring.txt"
node_positives_file = based_directory + "He_node_structure_positives_order_by_nodes.txt"
node_negatives_file = based_directory + "He_node_structure_negatives_order_by_nodes.txt"
label_structure_by_type_file = based_directory + "He_labels_structure_by_type_order_by_nodes.txt"
label_structure_file = based_directory + "He_labels_structure_order_by_nodes.txt"
label_content_by_type_file = based_directory + "He_labels_content_by_type_order_by_nodes.txt"
label_content_file = based_directory + "He_labels_content_order_by_nodes.txt"
result_directory = based_directory + "result_GCN_partAttribute_torch_coeffi/"
flag_by_type = False
flag_by_type_str = "bynoType"
modelName = 'GCN_partAttribute_torch_coeffi'

#
nodes_num = 807
#
outputm = 8
#
epochx = 600
#
seedn = 1

#0
#
def saveEmbedding(epochx):
    with open(
            result_directory+str(epochx)+"/" + modelName + '_embedding' + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(
                    outputm) + '.txt', "w", encoding='UTF-8') as f:
        print('Trainning time (Seconds) =' + str(end - start)+' loss='+str(loss), file=f)
        f.write('\n')
        for i in range(len(nodes_list)):
            node_i_embedding = str(nodes_list[i])
            for k in range(outputm):
                node_i_embedding = node_i_embedding + ',' + str(y_structure_embedding_list[i][k])
            f.writelines(node_i_embedding)
            f.write('\n')

    f.close()
#0
def application(epochx):
    #
    print("3.application(myCluster[structure]).....")
    myCluster(labels_structure_Count, y_structure_embedding_list, labels_structure, iters, result_directory+str(epochx)+"/", modelName+'_structureLabel', epochx, seedn, outputm)
    print("3.application(myClassification[structure]).....")
    myClassification(n_neighbors, y_structure_embedding_list, labels_structure, iters, split_classification, shuffle_classification, result_directory+str(epochx)+"/", modelName+'_structureLabel', epochx, seedn,outputm)
    structure_edgeEmbedding, structure_edgeLabel, structure_edge_count = nodeEmbedding_to_edgeEmbedding_by_multiplication(y_structure_embedding_list, edges_file,nodes_list, outputm)
    print("3.application(myLinkPrediction[structure]).....")
    for i in range(len(split_linkprediction)):
        myLinkPrediction(structure_edgeEmbedding, structure_edgeLabel, structure_edge_count, split_linkprediction[i], result_directory+str(epochx)+"/", modelName+'_structureLabel', epochx, seedn, outputm)

#############################################################
# 2.
#############################################################

def similarity1(a, b):
    n = len(a)
    s = 0
    for i in range(n):
        if a[i] == b[i] == 1:
            s = s + 1
    c = []
    for i in range(n):
        if a[i] == 1:
            c.append(a[i])
        else:
            c.append(b[i])
    max_s = sum(c)
    return s/max_s

#
def similarity2(a, b):
    a = list(set(a))
    b = list(set(b))
    n = len(a)
    s = 0
    for i in range(n):
        if a[i] == b[i] == 1:
            s = s + 1
    c = []
    for i in range(n):
        if a[i] == 1:
            c.append(a[i])
        elif b[i] == 1:
            c.append(b[i])
    max_s = sum(c)
    return s/max_s

def similarity_coefficient(a,b):
    n = len(a)
    s = 0
    for i in range(n):
        if a[i] == b[i] == 1:
            s = s + 1
    a_len = []
    for i in range(len(a)):
        if a[i] == 1:
            a_len.append(a[i])
    b_len=[]
    for i in range(len(b)):
        if b[i] == 1:
            b_len.append(b[i])
    max_s = sum(a_len)+sum(b_len)
    return 2*s / max_s


def get_xxxs_content_embedding_list(node_xxx_feature_file):
    xxxs_content_embedding_list = []
    xxxs_feature = []
    with open(node_xxx_feature_file, "r", encoding='UTF-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            node_adj_line = line.split('\t')
            xxx_feature = node_adj_line[4]
            feature_list = xxx_feature.split(";")
            for i in range(len(feature_list) - 1):
                if feature_list[i] not in xxxs_feature:
                    xxxs_feature.append(feature_list[i])
    fr.close()
    xxxs_feature_list = xxxs_feature
    xxxs_feature_list.sort()
    #print(xxxs_feature_list)
    l = len(xxxs_feature_list)
    print(l)
    with open(node_xxx_feature_file, "r", encoding='UTF-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            node_adj_line = line.split('\t')
            xxx_feature = node_adj_line[4]
            feature_list = xxx_feature.split(";")
            xxx_feature_embedding = init_xxx_feature_embedding(l)
            for i in range(len(feature_list) - 1):
                position = xxxs_feature_list.index(feature_list[i])
                xxx_feature_embedding[position] = 1
            xxxs_content_embedding_list.append(xxx_feature_embedding)
    fr.close()
    return xxxs_content_embedding_list
#
def data_load_one_adj(device):
    # one-hot
    node_list = []

    with open(nodes_file, "r", encoding='UTF-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            node_adj_line = line.split('\t')
            # print('node_adj_line=',node_adj_line[0])
            if node_adj_line[0] not in node_list:
                node_list.append(node_adj_line[0])

    fr.close()


    #
    xxxs_content_embedding_list = get_xxxs_content_embedding_list(nodes_file)

    node_zeros = np.zeros((len(node_list), len(node_list)))
    for i in range(len(node_list)):
        node_zeros[i][i] = node_zeros[i][i] + 1
    H = node_zeros
    #
    node_adj_matrix = np.zeros((len(node_list), len(node_list)))
    with open(edges_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split('\t')
            # print(node_adj_line)
            curr_node_x = node_adj_line[0]
            curr_node_y = node_adj_line[1]
            curr_node_x_position = node_list.index(curr_node_x)
            curr_node_y_position = node_list.index(curr_node_y)
            simlarity_value = similarity_coefficient(xxxs_content_embedding_list[curr_node_x_position], xxxs_content_embedding_list[curr_node_y_position])
            node_adj_matrix[curr_node_x_position][curr_node_y_position] = 1+simlarity_value
            node_adj_matrix[curr_node_y_position][curr_node_x_position] = 1+simlarity_value
            if simlarity_value!=0:
                print("simlarity_value=",simlarity_value)
    f.close()


    #

    for i in range(len(node_list)):
        node_adj_matrix[i][i] = 2.0

    A_hat = node_adj_matrix
    #
    D_hat = np.array(np.sum(node_adj_matrix, axis=0))  #
    D_hat = np.matrix(np.diag(D_hat))

    #
    D_hat_tensor = torch.tensor(D_hat).to(device)
    A_hat_tensor = torch.tensor(A_hat).to(device)
    H_tensor = torch.tensor(H).to(device)

    # DAD
    #DADH = D_hat ** -1 / 2 * A_hat * D_hat ** -1 / 2 * H
    DADH = D_hat_tensor.pow(-1/2)
    DADH[DADH == float("inf")] = 0
    DADH1 = torch.mm(DADH,A_hat_tensor.double())
    DADH1 = torch.mm(DADH1,DADH)
    DADH1 = torch.mm(DADH1,H_tensor.double())
    return DADH1, H, node_list

#2.4.0
def read_positive_from_file(node_positives_file,node_list):
    predict_positive_num = []
    label_positive_position = []
    with open(node_positives_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            for i in range(1, len(node_adj_line) - 1):
                label_x = node_adj_line[i]
                label_x_position = node_list.index(label_x)
                label_positive_position.append(label_x_position)
            predict_positive_num.append(len(node_adj_line) - 2)
    return predict_positive_num, label_positive_position
#2.4.1
def find_predict_label_positive(H, label_positive_position):
    label_positive=[]
    for label_x_position in label_positive_position:
        label_positive.append(H[label_x_position])
    return label_positive
#2.4.2
def predict_positive_cat(y,predict_positive_num):
    HGCN_output_split_tuple = y.split(1, 0)
    predict_positive_tensor = HGCN_output_split_tuple[0]
    for i in range(1, predict_positive_num[0]):
        predict_positive_tensor = torch.cat((predict_positive_tensor, HGCN_output_split_tuple[0]),0)
    for i in range(1, len(predict_positive_num)):
        for j in range(predict_positive_num[i]):
            predict_positive_tensor = torch.cat((predict_positive_tensor, HGCN_output_split_tuple[i]),0)
    return predict_positive_tensor

#2.5.0
def read_negative_from_file(node_negatives_file,node_list):
    predict_negative_num = []
    label_negative_position = []
    with open(node_negatives_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            negative_list = node_adj_line[1:]
            for i in range(0, len(negative_list) - 1):
                label_x = negative_list[i]
                label_x_position = node_list.index(label_x)
                label_negative_position.append(label_x_position)
            predict_negative_num.append(len(node_adj_line) - 2)
    return predict_negative_num, label_negative_position
#2.5.1
def find_predict_label_negative(H, label_negative_position):
    label_negative = []
    for label_x_position in range(len(label_negative_position)):
        label_negative.append(H[label_x_position])
    return label_negative
#2.5.2
def find_predict_label_negative_sample_1(H, predict_positive_num,predict_negative_num,label_negative_position):
    label_negative = []
    predict_negative_num_sample = []
    sum_negative = 0
    for k in range(len(predict_positive_num)):
        negative_num = min(predict_positive_num[k]*3,predict_negative_num[k])
        #print('predict_positive_num[k]*3=',predict_positive_num[k]*3)
        #print('predict_negative_num[k]=',predict_negative_num[k])
        #print('negative_num= ',negative_num)
        #print('sum_negative= ',sum_negative)
        #print('sum_negative + predict_negative_num[k] = ',sum_negative + predict_negative_num[k] )
        sample_list=random.sample(range(sum_negative, sum_negative + predict_negative_num[k]), negative_num)
        for i in sample_list:
            label_negative.append(H[label_negative_position[i]])
        predict_negative_num_sample.append(negative_num)
        sum_negative = sum_negative + predict_negative_num[k]

    return predict_negative_num_sample, label_negative
#2.5.3
def find_predict_label_negative_sample_2(H, node_list):
    predict_negative_num = []
    label_negative = []
    k = 0
    with open(node_negatives_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            node_adj_line = line.split(',')
            negative_list = node_adj_line[1:]
            negative_num_all = len(negative_list)
            negative_num = math.ceil(negative_num_all*0.8)
            sample_list = random.sample(range(0, negative_num_all - 1), negative_num)
            for i in sample_list:
                label_x = negative_list[i]
                label_x_position = node_list.index(label_x)
                label_negative.append(H[label_x_position])
            predict_negative_num.append(negative_num)
            k = k + 1
    return predict_negative_num, label_negative
#2.5.4
def predict_negative_cat(y,predict_negative_num):
    HGCN_output_split_tuple = y.split(1, 0)
    predict_negative_tensor = HGCN_output_split_tuple[0]
    for i in range(1, predict_negative_num[0]):
        predict_negative_tensor = torch.cat((predict_negative_tensor, HGCN_output_split_tuple[0]),0)
    for i in range(1, len(predict_negative_num)):
        for j in range(predict_negative_num[i]):
            predict_negative_tensor = torch.cat((predict_negative_tensor, HGCN_output_split_tuple[i]),0)
    return predict_negative_tensor

class HeGCN(nn.Module):
    def __init__(self):
        super(HeGCN, self).__init__()
        torch.manual_seed(seedn)
        self.linear_0 = nn.Linear(nodes_num, outputm)
        self.linear_1 = nn.Linear(outputm, nodes_num)
        init.xavier_uniform_(self.linear_0.weight)
        init.xavier_uniform_(self.linear_1.weight)

    def forward(self, x):
        x = F.relu(self.linear_0(x))
        x = self.linear_1(x)
        return x

features = []
def hook(module, inputx, output):
    features.append(output.clone().detach())
#############################################################
# 3.
#############################################################
#
labels_structureANDcontent_by_type,labels_kind = get_label_file(label_structure_by_type_file)
labels_structure, labels_structure_kind = get_label_file(label_structure_file)
labels_content, labels_content_kind = get_label_file(label_content_file)
labels_structure_Count = len(labels_structure_kind)
labels_content_Count = len(labels_content_kind)
iters = 10
split_classification = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
shuffle_classification =True
split_linkprediction = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
n_neighbors = 1
#
start = time.process_time()
if torch.cuda.is_available():
    print("GPU")
else:
    print("CPU")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("data loading now.....")
DADH, H, nodes_list = data_load_one_adj(device)
predict_positive_num, label_positive_position = read_positive_from_file(node_positives_file,nodes_list)
label_positive = find_predict_label_positive(H, label_positive_position)
predict_negative_num, label_negative_position = read_negative_from_file(node_negatives_file,nodes_list)
predict_negative_num_sample, label_negative = find_predict_label_negative_sample_1(H, predict_positive_num,predict_negative_num,label_negative_position)

print("data loading end.....")
DADH=DADH.clone().detach().requires_grad_(True).float()
net = HeGCN().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
crossentropyloss = nn.CrossEntropyLoss()

for i in range(1,epochx+1):
    handle = net.linear_0.register_forward_hook(hook)
    y = net(DADH)
    handle.remove()
    label_positive = find_predict_label_positive(H, label_positive_position)
    predict_positive = predict_positive_cat(y,predict_positive_num)
    label_positive = torch.tensor(label_positive)
    print('label_positive=', label_positive)
    label_positive = torch.topk(label_positive, 1)[1].squeeze(1)
    loss_positive = torch.mean(crossentropyloss(predict_positive, label_positive.to(device)))

    predict_negative_num_sample, label_negative =  find_predict_label_negative_sample_1(H, predict_positive_num, predict_negative_num, label_negative_position)
    predict_negative = predict_negative_cat(y, predict_negative_num_sample)
    label_negative= torch.tensor(label_negative)
    label_negative = torch.topk(label_negative, 1)[1].squeeze(1)
    loss_negative = torch.mean(crossentropyloss(predict_negative, label_negative.to(device)))

    loss = torch.sub(loss_positive, loss_negative)
    if i%600==0:
        print('loss=',loss)
        end = time.process_time()
        y_structure_embedding_list = features[i - 1].tolist()
        saveEmbedding(i)
        application(i)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




