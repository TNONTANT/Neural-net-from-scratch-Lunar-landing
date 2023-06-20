import pandas as pd
import numpy as np

# for finding value to calculate normalization
df = pd.read_csv("ce889_dataCollection.csv",header=None)
df.rename(columns={0:'pos_x',1:'pos_y',2:'vel_y',3:'vel_x'},inplace=True)

# data input x1
input_node1_list = df["pos_x"].values.tolist()
# data input x2
input_node2_list = df["pos_y"].values.tolist()
# data output y1
actual_output_node1_list = df["vel_x"].values.tolist()
# data output y2
actual_output_node2_list = df["vel_y"].values.tolist()

max_input1 = max(input_node1_list)
min_input1 = min(input_node1_list)

max_input2 = max(input_node2_list)
min_input2 = min(input_node2_list)

max_output1 = max(actual_output_node1_list)
min_output1 = min(actual_output_node1_list)

max_output2 = max(actual_output_node2_list)
min_output2 = min(actual_output_node2_list)

class NeuralNetHolder:
    def __init__(self):
        self.w_hid = pd.read_csv('w_hid.csv')
        self.w_out = pd.read_csv('w_out.csv')
        self.lamb = 1

    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity

        # sigmoid function
        def sigmoid(x):
            
            return 1.0 / (1.0 + (np.exp(-x*self.lamb)))

        input_list = input_row.split(",")
        
        inNode1 = float(input_list[0])
        inNode2 = float(input_list[1])

        # normalize input         
        inNode1 = (inNode1 - min_input1)/(max_input1 - min_input1)
        inNode2 = (inNode2 - min_input2)/(max_input2 - min_input2)
        
            
        hnode_size = int(len(self.w_hid)/2)
        hidd_node = [] # this list will be contain sigmoided out put node
        
        # input to hidden Layer
        
        for i in range(hnode_size):
        
            hNode = (inNode1 * self.w_hid[i]) + (inNode2 * self.w_hid[i+hnode_size])
            hNode = sigmoid(hNode)
        
            hidd_node.append(hNode)
            
        out_node = [0,0]
        for i in range(hnode_size):
            # output node 1 (unsigmoid)
            out_node[0] += (hidd_node[i]*self.w_out[i])
            # output node 2 (unsigmoid)
            out_node[1] += (hidd_node[i]*self.w_out[i+hnode_size])

        out_node = list(map(sigmoid,out_node)) 
        
        # de-Normalize to output
        X_Velocity = (out_node[0]*(max_output1 - min_output1)) + min_output1
        Y_Velocity = (out_node[1]*(max_output2 - min_output2)) + min_output2
                
        print([Y_Velocity, X_Velocity])
        
        return [X_Velocity, Y_Velocity]

