import pandas as pd
import numpy as np
import math

#######this file containing find ht ebest parameter including train and test data##############
###############################Preprocessing Data#################################

def pre_processing(df):
    # re columns name
    df.rename(columns={0:'pos_x',1:'pos_y',2:'vel_y',3:'vel_x'},inplace=True)
    # normalized data
    normalized_df=(df-df.min())/(df.max()-df.min())
    # data input x1
    lenght = normalized_df["pos_x"].values.tolist()

    # Partitioning Data to 70% for Training and 30% for Testing
    break_point = int(len(lenght)*0.7)

    # training set
    input_node1_list = normalized_df["pos_x"].values.tolist()[:break_point]
    input_node2_list = normalized_df["pos_y"].values.tolist()[:break_point]
    actual_output_node1_list = normalized_df["vel_x"].values.tolist()[:break_point]
    actual_output_node2_list = normalized_df["vel_y"].values.tolist()[:break_point]
    # testing set
    test_input_node1_list = normalized_df["pos_x"].values.tolist()[break_point:]
    test_input_node2_list = normalized_df["pos_y"].values.tolist()[break_point:]
    test_actual_output_node1_list = normalized_df["vel_x"].values.tolist()[break_point:]
    test_actual_output_node2_list = normalized_df["vel_y"].values.tolist()[break_point:]

    return input_node1_list, input_node2_list, actual_output_node1_list, actual_output_node2_list, test_input_node1_list, test_input_node2_list, test_actual_output_node1_list, test_actual_output_node2_list


###############################Function###########################################

# NeuralNetwork function (for Training Data)
def NeuralNetwork(epoch, lr, mt, node, inNode1_li, inNode2_li, actNode1_li, actNode2_li):
    
    # create intitial delta weight
    del_w_h = [0]*(node*2)
    del_w_o = [0]*(node*2)
    
    # create initial weight
    w_hid = np.random.uniform(-1,1,node*2)
    w_out = np.random.uniform(-1,1,node*2)
    
    lam = 1
    
    # activation function set to sigmoid
    def sigmoid(x):
        
        return 1.0 / (1.0 + np.exp(-x*lam))
        
    # list to store rmse and weight value
    rms_per_ep = []
    w_hid_ep = []
    w_out_ep = []

    # loop throught epoch
    for e in range(epoch):
        # storing data per epoch
        each_y_hat = [[],[]]
        each_rmse = []
        
        # loop through each row of data
        for row in range(len(inNode1_li)):
            print('Training Set epoch:', e, 'row', row)
            hnode_size = node
                    
            inNode1 = inNode1_li[row]
            inNode2 = inNode2_li[row]
            actNode1 = actNode1_li[row]
            actNode2 = actNode2_li[row]
                    
            #############################feed foword##################################
            # input to hidden Layer
            hidd_node = [] # this list will be contain sigmoided out put node
                        
            for i in range(hnode_size):
                        
                hNode = (inNode1 * w_hid[i]) + (inNode2 * w_hid[i+hnode_size])
                hNode = sigmoid(hNode)
                        
                hidd_node.append(hNode)
                    
            # hidden to output Layer
            out_node = [0,0]
            for i in range(hnode_size):
                # output node 1 (unsigmoid)
                out_node[0] += (hidd_node[i]*w_out[i])
                # output node 2 (unsigmoid)
                out_node[1] += (hidd_node[i]*w_out[i+hnode_size])
                        
            # now out_node is the list of sigmoided out_node (2 out node)
            out_node = list(map(sigmoid,out_node)) 
                
            # store y_hat data
            each_y_hat[0].append(out_node[0])
            each_y_hat[1].append(out_node[1])
                
            # error
            err = [[],[]]
            err[0] = actNode1 - out_node[0]
            err[1] = actNode2 - out_node[1]
                
            print(err)
                
            # rmse
            each_rmse.append(math.sqrt((err[0]**2 + err[1]**2)/2))
                        
            ##########################back propergation###############################
                        
            # local grdient for output Layer
            LG_output1 = lam * out_node[0] *(1 - out_node[0]) * err[0]
            LG_output2 = lam * out_node[1] *(1 - out_node[1]) * err[1]
                        
            # local grdient for hidden Layer
            LG_hidd = []
            for i in range(hnode_size):
                lg_hidd = lam*hidd_node[i]*(1-hidd_node[i])*((LG_output1*w_out[i])+(LG_output2*w_out[i+hnode_size]))
                LG_hidd.append(lg_hidd)
                
            # delta w of outputLayer
            delta_w_out1 = []
            delta_w_out2 = []
            for i in range(hnode_size):
                # delta w for out node 1
                deltaW1 = lr * LG_output1 * hidd_node[i] + (mt * del_w_o[i])
                # delta w for out node 1
                deltaW2 = lr * LG_output2 * hidd_node[i] + (mt * del_w_o[i+hnode_size])
                delta_w_out1.append(deltaW1)
                delta_w_out2.append(deltaW2)
                
            del_w_o = delta_w_out1 + delta_w_out2 #delta w o updated
                    
            delta_w_hidd1 = []
            delta_w_hidd2 = []
            for i in range(hnode_size):
                # delta w for hidd node 1
                deltaW1 = lr * LG_hidd[i] * inNode1 + (mt * del_w_h[i])
                # delta w for hidd node 1
                deltaW2 = lr * LG_hidd[i] * inNode2 + (mt * del_w_h[i+hnode_size])
                delta_w_hidd1.append(deltaW1)
                delta_w_hidd2.append(deltaW2)
                
            del_w_h = delta_w_hidd1 + delta_w_hidd2 #delta w h updated
                
            w_hid = [x + y for x, y in zip(w_hid, del_w_h)]
            w_out = [x + y for x, y in zip(w_out, del_w_o)]
            
        rms_per_ep.append(sum(each_rmse)/len(each_rmse))
        w_hid_ep.append(w_hid)
        w_out_ep.append(w_out)
        
    return rms_per_ep, each_y_hat, w_hid_ep, w_out_ep

# Neural network function for testing data
def TestNeural(epoch, lr, mt, node, inNode1_li, inNode2_li, actNode1_li, actNode2_li,
               train_w_hid_ep, train_w_out_ep, isParameter):
    lam = 1
    def sigmoid(x):
        
        return 1.0 / (1.0 + np.exp(-x*lam))
        
    rms_per_ep = []
    
    for e in range(epoch):
        # storing data per epoch
        each_rmse = []
        # for train and test ****************** TypeError: can't multiply sequence by non-int of type 'float'
        
        if isParameter:
        # for find the fit node
            w_hid = train_w_hid_ep
            w_out = train_w_out_ep
        
        else :
            w_hid = train_w_hid_ep[e]
            w_out = train_w_out_ep[e]

        
        for row in range(len(inNode1_li)):
            print('Test set epoch:', e, 'row', row)
            hnode_size = node
                    
            inNode1 = inNode1_li[row]
            inNode2 = inNode2_li[row]
            actNode1 = actNode1_li[row]
            actNode2 = actNode2_li[row]
            
                    
            #############################feed foword##################################
            # input to hidden Layer
            hidd_node = [] # this list will be contain sigmoided out put node
                        
            for i in range(hnode_size):
                        
                hNode = (inNode1 * w_hid[i]) + (inNode2 * w_hid[i+hnode_size])
                hNode = sigmoid(hNode)
                        
                hidd_node.append(hNode)
                    
            # hidden to output Layer
            out_node = [0,0]
            for i in range(hnode_size):
                # output node 1 (unsigmoid)
                out_node[0] += (hidd_node[i]*w_out[i])
                # output node 2 (unsigmoid)
                out_node[1] += (hidd_node[i]*w_out[i+hnode_size])
                        
            # now out_node is the list of sigmoided out_node (2 out node)
            out_node = list(map(sigmoid,out_node)) 
                
                
            # error
            err = [[],[]]
            err[0] = actNode1 - out_node[0]
            err[1] = actNode2 - out_node[1]
                
            print(err)
                
            # rmse
            each_rmse.append(math.sqrt((err[0]**2 + err[1]**2)/len(err)))
                        
        rms_per_ep.append(sum(each_rmse)/len(each_rmse))
        
    return rms_per_ep


###############################Find Best Parameter################################# 

# Node function for finding fit Node (fit number of network)
def Node(epoch,nodes, processed):
    # extract input
    # training set
    input_node1_list = processed[0]
    input_node2_list = processed[1]
    actual_output_node1_list = processed[2]
    actual_output_node2_list = processed[3]
    # test set
    test_input_node1_list = processed[4]
    test_input_node2_list = processed[5]
    test_actual_output_node1_list = processed[6]
    test_actual_output_node2_list = processed[7]
    rms_train = []
    rms_test = []
    bestnode = []
    lr = 0.01 # 1
    mt = 0    # 0.2
    for node in range(4,nodes): # set the node to begin at 4 to nodes(input)
        
        bestnode.append(node)
        
        Training = NeuralNetwork(epoch, lr, mt, node, input_node1_list, input_node2_list,
                                 actual_output_node1_list, actual_output_node2_list)
    
        train_rms_per_ep = Training[0][-1] #rms per node train
        train_w_hid_ep = Training[2][-1] #choose the last weight from every epoch of Training
        train_w_out_ep = Training[3][-1]

        rms_train.append(train_rms_per_ep)
        
        
        Testing = TestNeural(epoch, lr, mt, node, test_input_node1_list,test_input_node2_list,
                             test_actual_output_node1_list, test_actual_output_node2_list,
                             train_w_hid_ep, train_w_out_ep ,True)
        
        
        last_test_rms_per_ep = Testing[0]
        rms_test.append(last_test_rms_per_ep) # rms per node test

    return bestnode, rms_train, rms_test

# Learning Rate function for finding fit Learing rate parameter
def LearningRate(epoch, fnode, processed):
    # extract input
    # training set
    input_node1_list = processed[0]
    input_node2_list = processed[1]
    actual_output_node1_list = processed[2]
    actual_output_node2_list = processed[3]
    # test set
    test_input_node1_list = processed[4]
    test_input_node2_list = processed[5]
    test_actual_output_node1_list = processed[6]
    test_actual_output_node2_list = processed[7]

    rms_train = []
    rms_test = []
    mt = 0
    node = fnode # best node from node function
    LR = [0.05, 0.1, 0.15 , 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] # list of LR considered
    for lr in LR :
        
        Training = NeuralNetwork(epoch, lr, mt, node, input_node1_list, input_node2_list,
                                 actual_output_node1_list, actual_output_node2_list)

        train_rms_per_ep = Training[0][-1]
        train_w_hid_ep = Training[2][-1] #choose the last weight from every epoch of Training
        train_w_out_ep = Training[3][-1]
        
        rms_train.append(train_rms_per_ep)
        
        Testing = TestNeural(epoch, lr, mt, node, test_input_node1_list, test_input_node2_list,
                             test_actual_output_node1_list, test_actual_output_node2_list, 
                             train_w_hid_ep, train_w_out_ep, True)
        
        last_test_rms_per_ep = Testing[0]
        rms_test.append(last_test_rms_per_ep) # rms per node test
        
    return rms_train, rms_test, LR


# Momentum function for finding fit momentum parameter
def Momentum(epoch, fnode, flr, processed):
    # extract input
    # training set
    input_node1_list = processed[0]
    input_node2_list = processed[1]
    actual_output_node1_list = processed[2]
    actual_output_node2_list = processed[3]
    # test set
    test_input_node1_list = processed[4]
    test_input_node2_list = processed[5]
    test_actual_output_node1_list = processed[6]
    test_actual_output_node2_list = processed[7]
    rms_train = []
    rms_test = []
    MT = [0.05, 0.1, 0.15 , 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] # list of LR considered
    node = fnode # best node from node function
    lr = flr
    for mt in MT:
        
        Training = NeuralNetwork(epoch, lr, mt, node, input_node1_list,input_node2_list,
                                 actual_output_node1_list, actual_output_node2_list)

        train_rms_per_ep = Training[0][-1]
        train_w_hid_ep = Training[2][-1] #choose the last weight from every epoch of Training
        train_w_out_ep = Training[3][-1]
        
        rms_train.append(train_rms_per_ep)
        
        Testing = TestNeural(epoch, lr, mt, node, test_input_node1_list,test_input_node2_list,
                             test_actual_output_node1_list, test_actual_output_node2_list,
                             train_w_hid_ep, train_w_out_ep,True)
        
        last_test_rms_per_ep = Testing[0]
        rms_test.append(last_test_rms_per_ep) # rms per node test
        
    return rms_train, rms_test, MT



    
    
    
    
    