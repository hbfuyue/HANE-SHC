import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
import sklearn.metrics as Metric
from sklearn.cluster import DBSCAN
from sklearn.metrics import hamming_loss
from utils import *
##########################################################
#cluster
##########################################################
def myCluster(clusterCount, embedding, labels, iters, save_result_directory, modelName, epochx, seedn, outputm):
    estimator = KMeans(n_clusters=clusterCount)
    ARI_list = []
    NMI_list = []
    y_pred = []
    if iters:
        for i in range(iters):
            estimator.fit(embedding)
            y_pred = estimator.predict(embedding)
            nmi = normalized_mutual_info_score(labels, y_pred)
            NMI_list.append(nmi)
            ari = adjusted_rand_score(labels, y_pred)
            ARI_list.append(ari)
        nmi = sum(NMI_list) / len(NMI_list)
        ari = sum(ARI_list) / len(ARI_list)
        with open(save_result_directory+'myCluster_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
            print('NMI (10 avg): {:.4f} , ARI (10avg): {:.4f}'.format(nmi, ari), file=fw)
        fw.close()
        with open(save_result_directory+'pred_myCluster_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
            for i in range(len(y_pred)):
                print(y_pred[i], file=fw)
        fw.close()

##########################################################
#cluster2
##########################################################
def myCluster2(clusterCount_structure,clusterCount_content, embedding_structure,embedding_content, labels_structure, labels_content, iters, save_result_directory, modelName, epochx, seedn, outputm):
    estimator_structure = KMeans(n_clusters=clusterCount_structure)
    estimator_content = KMeans(n_clusters=clusterCount_content)
    ARI_list_structure = []
    NMI_list_structure = []
    ARI_list_content = []
    NMI_list_content = []
    ARI_list_structureANDcontent = []
    NMI_list_structureANDcontent = []
    y_pred_structure = []
    y_pred_content = []
    y_pred_content_modify = []
    if iters:
        for i in range(iters):
            estimator_structure.fit(embedding_structure)
            y_pred_structure = estimator_structure.predict(embedding_structure)
            #print('y_pred_structure',type(y_pred_structure))，
            estimator_content.fit(embedding_content)
            y_pred_content = estimator_content.predict(embedding_content)
            y_pred_content_modify = y_pred_content_modify_func(y_pred_content)
            labels_content_modify = labels_content_modify_func(labels_content)
            #
            nmi_structure = normalized_mutual_info_score(labels_structure,list(y_pred_structure))
            NMI_list_structure.append(nmi_structure)
            ari_structure = adjusted_rand_score(labels_structure,list(y_pred_structure))
            ARI_list_structure.append(ari_structure)

            #
            nmi_content = normalized_mutual_info_score(labels_content_modify,y_pred_content_modify)
            NMI_list_content.append(nmi_content)
            ari_content = adjusted_rand_score(labels_content_modify,y_pred_content_modify)
            ARI_list_content.append(ari_content)

            #
            nmi_structureANDcontent = normalized_mutual_info_score(labels_structure+labels_content_modify, list(y_pred_structure)+y_pred_content_modify)
            NMI_list_structureANDcontent.append(nmi_structureANDcontent)
            ari_structureANDcontent = adjusted_rand_score(labels_structure+labels_content_modify, list(y_pred_structure)+y_pred_content_modify)
            ARI_list_structureANDcontent.append(ari_structureANDcontent)

        #
        nmi_structure = sum(NMI_list_structure) / len(NMI_list_structure)
        ari_structure = sum(ARI_list_structure) / len(ARI_list_structure)
        #
        nmi_content = sum(NMI_list_content) / len(NMI_list_content)
        ari_content = sum(ARI_list_content) / len(ARI_list_content)
        #
        nmi_structureANDcontent = sum(NMI_list_structureANDcontent) / len(NMI_list_structureANDcontent)
        ari_structureANDcontent = sum(ARI_list_structureANDcontent) / len(ARI_list_structureANDcontent)
        with open(save_result_directory+'myCluster_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
            print('NMI (structure): {:.4f} , ARI (structure): {:.4f}'.format(nmi_structure, ari_structure), file=fw)
            print('NMI (content): {:.4f} , ARI (content): {:.4f}'.format(nmi_content, ari_content), file=fw)
            print('NMI (structureANDcontent): {:.4f} , ARI (structureANDcontent): {:.4f}'.format(nmi_structureANDcontent, ari_structureANDcontent), file=fw)
        fw.close()
        with open(save_result_directory+'pred_myCluster_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
            print('---structure---', file=fw)
            for i in range(len(y_pred_structure)):
                print(y_pred_structure[i], file=fw)
            print('---content_modify---', file=fw)
            for i in range(len(y_pred_content_modify)):
                print(y_pred_content_modify[i], file=fw)
            print('---content---', file=fw)
            for i in range(len(y_pred_content)):
                print(y_pred_content[i], file=fw)
        fw.close()


##########################################################
#DBSCAN
##########################################################
def myCluster_DBSCAN(eps_,min_samples_,embedding, labels, iters, save_result_directory, modelName, epochx, seedn, outputm):
    clustering = DBSCAN(eps=eps_, min_samples=min_samples_)
    ARI_list = []
    NMI_list = []
    y_pred = []
    if iters:
        for i in range(iters):
            clustering.fit(embedding)
            y_pred = clustering.labels_
            nmi = normalized_mutual_info_score(labels, y_pred)
            NMI_list.append(nmi)
            ari = adjusted_rand_score(labels, y_pred)
            ARI_list.append(ari)
        nmi = sum(NMI_list) / len(NMI_list)
        ari = sum(ARI_list) / len(ARI_list)
        with open(save_result_directory+'DBSCAN_myCluster_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
            print('NMI (10 avg): {:.4f} , ARI (10avg): {:.4f}'.format(nmi, ari), file=fw)
        fw.close()
        with open(save_result_directory+'DBSCAN_pred_myCluster_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
            for i in range(len(y_pred)):
                print(y_pred[i], file=fw)
        fw.close()




##########################################################
#myclassification
##########################################################
def myClassification(n_neighbors,embedding, labels, iters, split_list, shuffle, save_result_directory, modelName, epochx, seedn, outputm):
    embedding = np.array(embedding)
    labels = np.array(labels)
    print('label=',labels)
    for split in split_list:
        ss = split
        split = int(len(embedding) * split)
        micro_list = []
        macro_list = []
        y_pred = []
        if iters:
            for i in range(iters):
                if shuffle:
                    permutation = np.random.permutation(len(embedding))
                    embedding = embedding[permutation]
                    labels = labels[permutation]

                train_x = embedding[:split]
                test_x = embedding[split:]

                train_labels = labels[:split]
                test_labels = labels[split:]
                '''
                print("train_x=",train_x)
                print("test_x=", test_x)
                print("train_labels=", train_labels)
                print("test_labels=", test_labels)
                '''
                estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
                estimator.fit(train_x, train_labels)
                y_pred = estimator.predict(test_x)
                print("test_labels=",test_labels,len(test_labels))
                f1_macro = f1_score(test_labels, y_pred, average='macro')
                f1_micro = f1_score(test_labels, y_pred, average='micro')
                macro_list.append(f1_macro)
                micro_list.append(f1_micro)
            with open(save_result_directory+'myClassification_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
                print('KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(
                iters, ss, n_neighbors, sum(macro_list) / len(macro_list), sum(micro_list) / len(micro_list)), file=fw)
            fw.close()
            with open(save_result_directory+'pred_myClassification_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw2:
                for i in range(len(y_pred)):
                    print(y_pred[i], file=fw2)
            fw2.close()

##########################################################
#myclassification1_2
##########################################################
def myClassification1_2(n_neighbors,embedding, labels, iters, split_list, shuffle, save_result_directory, modelName, epochx, seedn, outputm):
    embedding = np.array(embedding)
    labels = np.array(labels)
    for split in split_list:
        ss = split
        split = int(len(embedding) * split)
        micro_list = []
        macro_list = []
        y_pred = []
        if iters:
            for i in range(iters):
                if shuffle:
                    permutation = np.random.permutation(len(embedding))
                    embedding = embedding[permutation]
                    labels = labels[permutation]

                train_x = embedding
                test_x = embedding

                train_labels = labels
                test_labels = labels
                print("train_x=", train_x)
                print("test_x=", test_x)
                print("train_labels=", train_labels)
                print("test_labels=", test_labels)
                estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
                estimator.fit(train_x, train_labels)
                y_pred = estimator.predict(test_x)
                f1_macro = f1_score(test_labels, y_pred, average='macro')
                f1_micro = f1_score(test_labels, y_pred, average='micro')
                macro_list.append(f1_macro)
                micro_list.append(f1_micro)
            with open(save_result_directory+'myClassification_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
                print('KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(
                iters, ss, n_neighbors, sum(macro_list) / len(macro_list), sum(micro_list) / len(micro_list)), file=fw)
            fw.close()
            with open(save_result_directory+'pred_myClassification_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
                for i in range(len(y_pred)):
                    print(y_pred[i], file=fw)
            fw.close()


##########################################################
#myclassification2
##########################################################
def myClassification2(n_neighbors_structure, n_neighbors_content,embedding_structure, embedding_content, labels_structure, labels_content, iters, split_list, shuffle, save_result_directory, modelName, epochx, seedn, outputm):
    embedding_structure_array = np.array(embedding_structure)
    labels_structure_array = np.array(labels_structure)
    embedding_content_array = np.array(embedding_content)
    labels_content_array = np.array(labels_content)
    for split in split_list:
        ss = split
        split = int(len(embedding_structure) * split)
        micro_list_structure = []
        macro_list_structure = []
        micro_list_content = []
        macro_list_content = []
        micro_list_structureANDcontent = []
        macro_list_structureANDcontent = []
        y_pred_structure = []
        y_pred_content = []
        if iters:
            for i in range(iters):
                if shuffle:
                    permutation = np.random.permutation(len(embedding_structure_array))
                    #
                    embedding_structure_array = embedding_structure_array[permutation]
                    labels_structure_array = labels_structure_array[permutation]
                    #
                    embedding_content_array = embedding_content_array[permutation]
                    labels_content_array = labels_content_array[permutation]

                #
                train_x_structure = embedding_structure_array[:split]
                test_x_structure = embedding_structure_array[split:]
                #
                train_x_content = embedding_content_array[:split]
                test_x_content = embedding_content_array[split:]

                #
                train_labels_structure = labels_structure_array[:split]
                test_labels_structure = labels_structure_array[split:]
                #
                train_labels_content = labels_content_array[:split]
                test_labels_content = labels_content_array[split:]

                #
                estimator_structure = KNeighborsClassifier(n_neighbors=n_neighbors_structure)
                estimator_structure.fit(train_x_structure, train_labels_structure)
                y_pred_structure = estimator_structure.predict(test_x_structure)

                #
                estimator_content = KNeighborsClassifier(n_neighbors=n_neighbors_content)
                estimator_content.fit(train_x_content, train_labels_content)
                y_pred_content = estimator_content.predict(test_x_content)

                #
                f1_macro_structure = f1_score(list(test_labels_structure), list(y_pred_structure), average='macro')
                f1_micro_structure = f1_score(list(test_labels_structure), list(y_pred_structure), average='micro')
                macro_list_structure.append(f1_macro_structure)
                micro_list_structure.append(f1_micro_structure)
                #
                f1_macro_content = f1_score(list(test_labels_content),list(y_pred_content), average='macro')
                f1_micro_content = f1_score(list(test_labels_content),list(y_pred_content), average='micro')
                macro_list_content.append(f1_macro_content)
                micro_list_content.append(f1_micro_content)

                #
                f1_macro_structureANDcontent = f1_score(list(test_labels_structure)+list(test_labels_content), list(y_pred_structure)+list(y_pred_content), average='macro')
                f1_micro_structureANDcontent = f1_score(list(test_labels_structure)+list(test_labels_content), list(y_pred_structure)+list(y_pred_content), average='micro')
                macro_list_structureANDcontent.append(f1_macro_structureANDcontent)
                micro_list_structureANDcontent.append(f1_micro_structureANDcontent)
            with open(save_result_directory+'myClassification_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
                print('[structure]KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(iters, ss, n_neighbors_structure, sum(macro_list_structure) / len(macro_list_structure), sum(micro_list_structure) / len(micro_list_structure)), file=fw)
                print('[content]KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(iters, ss, n_neighbors_content,sum(macro_list_content) / len(macro_list_content),sum(micro_list_content) / len(micro_list_content)),file=fw)
                print('[structureANDcontent]KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(iters, ss,n_neighbors_structure + n_neighbors_content,sum(macro_list_structureANDcontent) / len(macro_list_structureANDcontent),sum(micro_list_structureANDcontent) / len(micro_list_structureANDcontent)),file=fw)
            fw.close()
            with open(save_result_directory+'pred_myClassification_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
                for i in range(len(y_pred_structure)):
                    print(y_pred_structure[i], file=fw)
            fw.close()

##########################################################
#myclassification2_2
##########################################################
def myClassification2_2(n_neighbors_structure, n_neighbors_content,embedding_structure, embedding_content, labels_structure, labels_content, iters, split_list, shuffle, save_result_directory, modelName, epochx, seedn, outputm):
    embedding_structure_array = np.array(embedding_structure)
    labels_structure_array = np.array(labels_structure)
    embedding_content_array = np.array(embedding_content)
    labels_content_array = np.array(labels_content)
    for split in split_list:
        ss = split
        split = int(len(embedding_structure) * split)
        micro_list_structure = []
        macro_list_structure = []
        micro_list_content = []
        macro_list_content = []
        micro_list_structureANDcontent = []
        macro_list_structureANDcontent = []
        y_pred_structure = []
        y_pred_content = []
        if iters:
            for i in range(iters):
                if shuffle:
                    permutation = np.random.permutation(len(embedding_structure_array))
                    #
                    embedding_structure_array = embedding_structure_array[permutation]
                    labels_structure_array = labels_structure_array[permutation]
                    #
                    embedding_content_array = embedding_content_array[permutation]
                    labels_content_array = labels_content_array[permutation]

                #
                train_x_structure = embedding_structure_array
                test_x_structure = embedding_structure_array
                #
                train_x_content = embedding_content_array
                test_x_content = embedding_content_array

                #
                train_labels_structure = labels_structure_array
                test_labels_structure = labels_structure_array
                #
                train_labels_content = labels_content_array
                test_labels_content = labels_content_array

                #
                estimator_structure = KNeighborsClassifier(n_neighbors=n_neighbors_structure)
                estimator_structure.fit(train_x_structure, train_labels_structure)
                y_pred_structure = estimator_structure.predict(test_x_structure)

                #
                estimator_content = KNeighborsClassifier(n_neighbors=n_neighbors_content)
                estimator_content.fit(train_x_content, train_labels_content)
                y_pred_content = estimator_content.predict(test_x_content)

                #
                f1_macro_structure = f1_score(list(test_labels_structure), list(y_pred_structure), average='macro')
                f1_micro_structure = f1_score(list(test_labels_structure), list(y_pred_structure), average='micro')
                macro_list_structure.append(f1_macro_structure)
                micro_list_structure.append(f1_micro_structure)
                #
                f1_macro_content = f1_score(list(test_labels_content),list(y_pred_content), average='macro')
                f1_micro_content = f1_score(list(test_labels_content),list(y_pred_content), average='micro')
                macro_list_content.append(f1_macro_content)
                micro_list_content.append(f1_micro_content)

                #
                f1_macro_structureANDcontent = f1_score(list(test_labels_structure)+list(test_labels_content), list(y_pred_structure)+list(y_pred_content), average='macro')
                f1_micro_structureANDcontent = f1_score(list(test_labels_structure)+list(test_labels_content), list(y_pred_structure)+list(y_pred_content), average='micro')
                macro_list_structureANDcontent.append(f1_macro_structureANDcontent)
                micro_list_structureANDcontent.append(f1_micro_structureANDcontent)
            with open(save_result_directory+'myClassification_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
                print('[structure]KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(iters, ss, n_neighbors_structure, sum(macro_list_structure) / len(macro_list_structure), sum(micro_list_structure) / len(micro_list_structure)), file=fw)
                print('[content]KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(iters, ss, n_neighbors_content,sum(macro_list_content) / len(macro_list_content),sum(micro_list_content) / len(micro_list_content)),file=fw)
                print('[structureANDcontent]KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(iters, ss,n_neighbors_structure + n_neighbors_content,sum(macro_list_structureANDcontent) / len(macro_list_structureANDcontent),sum(micro_list_structureANDcontent) / len(micro_list_structureANDcontent)),file=fw)
            fw.close()
            with open(save_result_directory+'pred_myClassification_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
                for i in range(len(y_pred_structure)):
                    print(y_pred_structure[i], file=fw)
            fw.close()

##########################################################
#myclassification2_3
##########################################################
def myClassification2_3(n_neighbors_structure, n_neighbors_content,embedding_structure_1, embedding_structure_2, embedding_content, labels_structure, labels_content, iters, split_num, save_result_directory_1,save_result_directory_2, modelName1,modelName2, epochx, seedn, outputm):
    embedding_structure_array_1 = np.array(embedding_structure_1)
    embedding_structure_array_2 = np.array(embedding_structure_2)
    labels_structure_array = np.array(labels_structure)
    embedding_content_array = np.array(embedding_content)
    labels_content_array = np.array(labels_content)
    split = int(len(embedding_structure_1) * split_num)
    micro_list_structure_1 = []
    macro_list_structure_1 = []

    micro_list_structure_2 = []
    macro_list_structure_2 = []
    micro_list_content = []
    macro_list_content = []

    micro_list_structureANDcontent_1 = []
    macro_list_structureANDcontent_1 = []
    micro_list_structureANDcontent_2 = []
    macro_list_structureANDcontent_2 = []
    y_pred_structure_1 = []
    y_pred_structure_2 = []
    #
    train_x_structure_1 = embedding_structure_array_1[:split]
    test_x_structure_1 = embedding_structure_array_1[split:]
    train_x_structure_2 = embedding_structure_array_2[:split]
    test_x_structure_2 = embedding_structure_array_2[split:]
    #
    train_x_content = embedding_content_array[:split]
    test_x_content = embedding_content_array[split:]


    #
    train_labels_structure = labels_structure_array[:split]
    test_labels_structure = labels_structure_array[split:]
    #
    train_labels_content = labels_content_array[:split]
    test_labels_content = labels_content_array[split:]

    #
    estimator_structure_1 = KNeighborsClassifier(n_neighbors=n_neighbors_structure)
    estimator_structure_1.fit(train_x_structure_1, train_labels_structure)
    y_pred_structure_1 = estimator_structure_1.predict(test_x_structure_1)

    estimator_structure_2 = KNeighborsClassifier(n_neighbors=n_neighbors_structure)
    estimator_structure_2.fit(train_x_structure_2, train_labels_structure)
    y_pred_structure_2 = estimator_structure_2.predict(test_x_structure_2)

    #
    estimator_content = KNeighborsClassifier(n_neighbors=n_neighbors_content)
    estimator_content.fit(train_x_content, train_labels_content)
    y_pred_content = estimator_content.predict(test_x_content)

    print('list(test_labels_structure)=',len(list(test_labels_structure)))
    print('list(y_pred_structure_1)=', len(list(y_pred_structure_1)))
    #
    f1_macro_structure_1 = f1_score(list(test_labels_structure), list(y_pred_structure_1), average='macro')
    f1_micro_structure_1 = f1_score(list(test_labels_structure), list(y_pred_structure_1), average='micro')
    macro_list_structure_1.append(f1_macro_structure_1)
    micro_list_structure_1.append(f1_micro_structure_1)

    f1_macro_structure_2 = f1_score(list(test_labels_structure), list(y_pred_structure_2), average='macro')
    f1_micro_structure_2 = f1_score(list(test_labels_structure), list(y_pred_structure_2), average='micro')
    macro_list_structure_2.append(f1_macro_structure_2)
    micro_list_structure_2.append(f1_micro_structure_2)
    #
    f1_macro_content = f1_score(list(test_labels_content), list(y_pred_content), average='macro')
    f1_micro_content = f1_score(list(test_labels_content), list(y_pred_content), average='micro')
    macro_list_content.append(f1_macro_content)
    micro_list_content.append(f1_micro_content)

    #
    f1_macro_structureANDcontent_1 = f1_score(list(test_labels_structure) + list(test_labels_content),
                                            list(y_pred_structure_1) + list(y_pred_content), average='macro')
    f1_micro_structureANDcontent_1 = f1_score(list(test_labels_structure) + list(test_labels_content),
                                            list(y_pred_structure_1) + list(y_pred_content), average='micro')
    macro_list_structureANDcontent_1.append(f1_macro_structureANDcontent_1)
    micro_list_structureANDcontent_1.append(f1_micro_structureANDcontent_1)

    f1_macro_structureANDcontent_2 = f1_score(list(test_labels_structure) + list(test_labels_content),
                                            list(y_pred_structure_2) + list(y_pred_content), average='macro')
    f1_micro_structureANDcontent_2 = f1_score(list(test_labels_structure) + list(test_labels_content),
                                            list(y_pred_structure_2) + list(y_pred_content), average='micro')
    macro_list_structureANDcontent_2.append(f1_macro_structureANDcontent_2)
    micro_list_structureANDcontent_2.append(f1_micro_structureANDcontent_2)


    with open(save_result_directory_1+'myClassification_result_'+ modelName1 + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
        print('[structure]KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(iters, split_num, n_neighbors_structure, sum(macro_list_structure_1) / len(macro_list_structure_1), sum(micro_list_structure_1) / len(micro_list_structure_1)), file=fw)
        print('[content]KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(iters, split_num, n_neighbors_content,sum(macro_list_content) / len(macro_list_content),sum(micro_list_content) / len(micro_list_content)),file=fw)
        print('[structureANDcontent]KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(iters, split_num,n_neighbors_structure + n_neighbors_content,sum(macro_list_structureANDcontent_1) / len(macro_list_structureANDcontent_1),sum(micro_list_structureANDcontent_1) / len(micro_list_structureANDcontent_1)),file=fw)
    fw.close()
    with open(save_result_directory_2+'myClassification_result_'+ modelName2 + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
        print('[structure]KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(iters, split_num, n_neighbors_structure, sum(macro_list_structure_2) / len(macro_list_structure_2), sum(micro_list_structure_2) / len(micro_list_structure_2)), file=fw)
        print('[content]KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(iters, split_num, n_neighbors_content,sum(macro_list_content) / len(macro_list_content),sum(micro_list_content) / len(micro_list_content)),file=fw)
        print('[structureANDcontent]KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(iters, split_num,n_neighbors_structure + n_neighbors_content,sum(macro_list_structureANDcontent_2) / len(macro_list_structureANDcontent_2),sum(micro_list_structureANDcontent_2) / len(micro_list_structureANDcontent_2)),file=fw)
    fw.close()


##########################################################
#myLinkPrediction
##########################################################
def myLinkPrediction(edgeEmbedding, edgeLabel, edge_count, split, save_result_directory, modelName, epochx, seedn, outputm):


    train_positive_counts = 0
    train_negative_counts = 0

    embedding_train = []
    embedding_test = []
    label_train = []
    label_test = []
    for i in range(len(edgeLabel)):
        if edgeLabel[i]==1:
            if train_positive_counts <= int(edge_count*split):
                train_positive_counts = train_positive_counts + 1
                embedding_train.append(edgeEmbedding[i])
                label_train.append(edgeLabel[i])
            else:
                embedding_test.append(edgeEmbedding[i])
                label_test.append(edgeLabel[i])
        else:
            if train_negative_counts <= int(edge_count*split):
                train_negative_counts = train_negative_counts + 1
                embedding_train.append(edgeEmbedding[i])
                label_train.append(edgeLabel[i])
            else:
                embedding_test.append(edgeEmbedding[i])
                label_test.append(edgeLabel[i])

    embedding_train = np.array(embedding_train)
    #label_train = np.array(label_train)
    embedding_test = np.array(embedding_test)
    #label_test = np.array(label_test)
    #print('label_train',label_train)
    #print('label_test', label_test)
    lgr = linear_model.LogisticRegression(C=1.0, penalty='l2', random_state=1)
    lgr.fit(embedding_train, label_train)
    predict_train = list(lgr.predict(embedding_train))
    #print('predict_train')
    #print(list(predict_train))
    AUC_score_train = Metric.roc_auc_score(label_train, predict_train)
    F1_score_train = f1_score(label_train, predict_train)
    predict_test = lgr.predict(embedding_test)
    #print('predict_test')
    #print(list(predict_test))
    AUC_score_test = Metric.roc_auc_score(label_test, list(predict_test))
    F1_score_test = f1_score(label_test, list(predict_test))
    with open(save_result_directory+'myLinkPrediction_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
        print('LinkPrediction---train({}): (AUC: {:.4f}; f1: {:.4f}) LinkPrediction---test: (AUC: {:.4f}; f1: {:.4f})'.format(split,AUC_score_train, F1_score_train, AUC_score_test, F1_score_test), file=fw)
    fw.close()
    return label_train, predict_train, label_test, predict_test

##########################################################
#myLinkPrediction1_2是对myLinkPrediction的修改，对全部数据进行训练和测试
##########################################################
def myLinkPrediction1_2(edgeEmbedding, edgeLabel, edge_count, split, save_result_directory, modelName, epochx, seedn, outputm):


    train_positive_counts = 0
    train_negative_counts = 0

    embedding_train = []
    embedding_test = []
    label_train = []
    label_test = []
    for i in range(len(edgeLabel)):
        if edgeLabel[i]==1:
            if train_positive_counts <= int(edge_count*split):
                train_positive_counts = train_positive_counts + 1
                embedding_train.append(edgeEmbedding[i])
                label_train.append(edgeLabel[i])
            else:
                embedding_test.append(edgeEmbedding[i])
                label_test.append(edgeLabel[i])
        else:
            if train_negative_counts <= int(edge_count*split):
                train_negative_counts = train_negative_counts + 1
                embedding_train.append(edgeEmbedding[i])
                label_train.append(edgeLabel[i])
            else:
                embedding_test.append(edgeEmbedding[i])
                label_test.append(edgeLabel[i])

    embedding_train_new = np.array(embedding_train+embedding_test)
    embedding_test_new = np.array(embedding_train+embedding_test)
    label_train_new = label_train + label_test
    label_test_new = label_train + label_test

    lgr = linear_model.LogisticRegression(C=1.0, penalty='l2', random_state=1)
    lgr.fit(embedding_train_new, label_train_new)
    predict_train = list(lgr.predict(embedding_train_new))
    print('predict_train')
    print(list(predict_train))
    AUC_score_train = Metric.roc_auc_score(label_train_new, predict_train)
    F1_score_train = f1_score(label_train_new, predict_train)
    predict_test = lgr.predict(embedding_test_new)
    print('predict_test')
    print(list(predict_test))
    AUC_score_test = Metric.roc_auc_score(label_test_new, list(predict_test))
    F1_score_test = f1_score(label_test_new, list(predict_test))
    with open(save_result_directory+'myLinkPrediction_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
        print('LinkPrediction---train: (AUC: {:.4f}; f1: {:.4f}) LinkPrediction---test: (AUC: {:.4f}; f1: {:.4f})'.format(AUC_score_train, F1_score_train, AUC_score_test, F1_score_test), file=fw)
    fw.close()
    return label_train_new, predict_train, label_test_new, predict_test

##########################################################
#myLinkPrediction2
# ##########################################################
def myLinkPrediction2(structure_label_train_array, content_label_train_array, structure_predict_train, content_predict_train, structure_label_test_array, content_label_test_array, structure_predict_test, content_predict_test, save_result_directory, modelName, epochx, seedn, outputm):


    structureANDcontent_AUC_score_train = Metric.roc_auc_score(list(structure_label_train_array)+list(content_label_train_array), list(structure_predict_train)+list(content_predict_train))
    structureANDcontent_F1_score_train = f1_score(list(structure_label_train_array)+list(content_label_train_array), list(structure_predict_train)+list(content_predict_train))
    structureANDcontent_AUC_score_test = Metric.roc_auc_score(list(structure_label_test_array)+list(content_label_test_array), list(structure_predict_test)+list(content_predict_test))
    structureANDcontent_F1_score_test = f1_score(list(structure_label_test_array)+list(content_label_test_array), list(structure_predict_test)+list(content_predict_test))


    with open(save_result_directory+'myLinkPrediction_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
        print('---structureANDcontent---', file=fw)
        print('LinkPrediction---train: (AUC: {:.4f}; f1: {:.4f}) LinkPrediction---test: (AUC: {:.4f}; f1: {:.4f})'.format(structureANDcontent_AUC_score_train, structureANDcontent_F1_score_train, structureANDcontent_AUC_score_test, structureANDcontent_F1_score_test),file=fw)
    fw.close()

##########################################################
#myLinkPrediction3
# ##########################################################
def myLinkPrediction3(structure_label_train_array, content_label_train_array, structure_predict_train, content_predict_train, structure_label_test_array, content_label_test_array, structure_predict_test, content_predict_test, save_result_directory, modelName, epochx, seedn, outputm):


    #
    structure_label_train_array_to_list = list(structure_label_train_array)
    content_label_train_array_to_list = list(content_label_train_array)
    structure_predict_train_to_list = list(structure_predict_train)
    content_predict_train_to_list = list(content_predict_train)
    structure_label_test_array_to_list = list(structure_label_test_array)
    content_label_test_array_to_list = list(content_label_test_array)
    structure_predict_test_to_list = list(structure_predict_test)
    content_predict_test_to_list = list(content_predict_test)

    #
    print("process before")
    print("structure_label_train_array_to_list=",len(structure_label_train_array_to_list),structure_label_train_array_to_list)
    print("  content_label_train_array_to_list=",len(content_label_train_array_to_list),content_label_train_array_to_list)
    print("    structure_predict_train_to_list=",len(structure_predict_train_to_list),structure_predict_train_to_list)
    print("      content_predict_train_to_list=",len(content_predict_train_to_list),content_predict_train_to_list)
    print(" structure_label_test_array_to_list=",len(structure_label_test_array_to_list),structure_label_test_array_to_list)
    print("   content_label_test_array_to_list=",len(content_label_test_array_to_list),content_label_test_array_to_list)
    print("     structure_predict_test_to_list=",len(structure_predict_test_to_list),structure_predict_test_to_list)
    print("       content_predict_test_to_list=",len(content_predict_test_to_list),content_predict_test_to_list)

    for i in range(len(structure_label_train_array_to_list)):
        if content_predict_train_to_list[i] == structure_label_train_array_to_list[i]:
            structure_predict_train_to_list[i] = content_predict_train_to_list[i]
    for i in range(len(structure_label_test_array_to_list)):
        if content_predict_test_to_list[i] == structure_label_test_array_to_list[i]:
            structure_predict_test_to_list[i] = content_predict_test_to_list[i]

    print("process end")
    print("structure_label_train_array_to_list=",len(structure_label_train_array_to_list),structure_label_train_array_to_list)
    print("  content_label_train_array_to_list=",len(content_label_train_array_to_list),content_label_train_array_to_list)
    print("    structure_predict_train_to_list=",len(structure_predict_train_to_list),structure_predict_train_to_list)
    print("      content_predict_train_to_list=",len(content_predict_train_to_list),content_predict_train_to_list)
    print(" structure_label_test_array_to_list=",len(structure_label_test_array_to_list),structure_label_test_array_to_list)
    print("   content_label_test_array_to_list=",len(content_label_test_array_to_list),content_label_test_array_to_list)
    print("     structure_predict_test_to_list=",len(structure_predict_test_to_list),structure_predict_test_to_list)
    print("       content_predict_test_to_list=",len(content_predict_test_to_list),content_predict_test_to_list)

    #
    '''
    structureANDcontent_AUC_score_train = Metric.roc_auc_score(structure_label_train_array_to_list+content_label_train_array_to_list, structure_predict_train_to_list+content_predict_train_to_list)
    structureANDcontent_F1_score_train = f1_score(structure_label_train_array_to_list+content_label_train_array_to_list, structure_predict_train_to_list+content_predict_train_to_list)
    structureANDcontent_AUC_score_test = Metric.roc_auc_score(structure_label_test_array_to_list+content_label_test_array_to_list, structure_predict_test_to_list+content_predict_test_to_list)
    structureANDcontent_F1_score_test = f1_score(structure_label_test_array_to_list+content_label_test_array_to_list, structure_predict_test_to_list+content_predict_test_to_list)
    '''
    structureANDcontent_AUC_score_train = Metric.roc_auc_score(structure_label_train_array_to_list,structure_predict_train_to_list)
    structureANDcontent_F1_score_train = f1_score(structure_label_train_array_to_list,structure_predict_train_to_list)
    structureANDcontent_AUC_score_test = Metric.roc_auc_score(structure_label_test_array_to_list,structure_predict_test_to_list)
    structureANDcontent_F1_score_test = f1_score(structure_label_test_array_to_list,structure_predict_test_to_list)

    with open(save_result_directory+'myLinkPrediction_result_'+ modelName + '_epochx' + str(epochx) + '_seedn' + str(seedn) + '_outputm' + str(outputm) +'.txt', 'a') as fw:
        print('---structureANDcontent---', file=fw)
        print('LinkPrediction---train: (AUC: {:.4f}; f1: {:.4f}) LinkPrediction---test: (AUC: {:.4f}; f1: {:.4f})'.format(structureANDcontent_AUC_score_train, structureANDcontent_F1_score_train, structureANDcontent_AUC_score_test, structureANDcontent_F1_score_test),file=fw)
    fw.close()




