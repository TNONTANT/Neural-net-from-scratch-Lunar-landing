import pandas as pd
import numpy as np

#df = pd.read_csv("Data1.csv",header=None)
df = pd.read_csv("ce889_dataCollection.csv",header=None)
#df.rename(columns={0:'pos_x',1:'pos_y',2:'vel_x',3:'vel_y'},inplace=True)
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


# My best weight from individual.py put the weight below

w_hid = [-5.026065835935741, -3.153711409952522, -0.8779406075474772, -7.476141866090503, -6.550454980898151, -2.138742513249982, -5.033343359311306, 4.259227735404377, -5.41454828996576, -0.7376018689287258, -2.482143963777884, -1.9954331027176202, -0.595636391846447, -0.7624710867830837, -1.0830509587208352, -6.361959509436548, -2.605920899709469, -1.1941096210069349, -5.032153564017538, -5.018480163374129, -10.3129442325473, -0.93068800611968, -4.994474721993648, -7.682060778977345, -3.801878315484222, -0.6195887516565798, -4.341425651924229, -4.374870607092168, 3.2497315777624625, -2.651875937825664, -5.431953302018205, -6.604609975100675, 2.2651365900235865, -3.4430597669734837, -5.822824311290423, -2.450996248405162, -6.419168766353122, -3.205836792481171, -6.791178674430602, -5.628784346517042, -5.858537555071579, -6.930577085435427, -6.756381122865463, -6.311568941483945, -3.390273378381575, -5.716306008596388, -6.281392204347392, -2.4831777027394257, -4.952448697815352, -0.7010461257629215, -6.537092811563485, -3.428721768203662, -3.2839036620096476, -5.2715557721106165, -6.947412905080696, -4.7304450385893775, -4.639551573540551, -16.720540824235922]
w_out = [-1.6240174918727674, -2.114194997150552, -0.980964503752461, -1.3993168608732522, 3.9205803402207247, -2.187053874294857, -1.420901178883012, 1.8550472725452074, 1.7156161225618722, -0.4978325151766068, -1.8938226371469322, -2.07305757227269, 0.5502697123446204, -0.6074923038532999, -1.0167754903711939, 3.5402742807002734, -2.4582080658156014, -1.4906159498121694, -1.4778555654767205, 0.20445497118839195, 4.54530765480573, -1.0852941429905318, -0.6534002125786021, 4.421788463279491, -2.7556470373788553, 0.2875341159108888, -2.5887397611457406, -1.493758041594639, 3.8465753012109274, -3.114556418287462, 0.5347253008917048, 3.7534929354752, -1.7135610546497269, -1.3377892425522802, 1.8516761096551395, -2.9122351218582363, -2.1836718788278593, -2.8646209537257183, 4.119362658500694, 1.2968686066903774, 1.9971088826906713, 3.9215950194750353, 4.056279590631355, 3.0055432421688075, -1.5576090907859053, 1.3373238444365254, 3.0791566548376914, -2.942786878224495, -1.6971687303728447, 3.7981127005424122, 3.6000214411450795, -2.700678927428118, 0.2500955819526525, -0.24913422565099252, 4.23958659540116, -1.1704634390773676, -1.195010712860606, -8.297034202768632]

lamb = 1

class NeuralNetHolder:
    def __init__(self):
        super().__init__()

    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity

        # sigmoid function
        def sigmoid(x):
            
            return 1.0 / (1.0 + (np.exp(-x*lamb)))

        input_list = input_row.split(",")
        
        inNode1 = float(input_list[0])
        inNode2 = float(input_list[1])

        # normalize input         
        inNode1 = (inNode1 - min_input1)/(max_input1 - min_input1)
        inNode2 = (inNode2 - min_input2)/(max_input2 - min_input2)
        
            
        hnode_size = int(len(w_hid)/2)
        hidd_node = [] # this list will be contain sigmoided out put node
        
        # input to hidden Layer
        
        for i in range(hnode_size):
        
            hNode = (inNode1 * w_hid[i]) + (inNode2 * w_hid[i+hnode_size])
            hNode = sigmoid(hNode)
        
            hidd_node.append(hNode)
            
        out_node = [0,0]
        for i in range(hnode_size):
            # output node 1 (unsigmoid)
            out_node[0] += (hidd_node[i]*w_out[i])
            # output node 2 (unsigmoid)
            out_node[1] += (hidd_node[i]*w_out[i+hnode_size])

        out_node = list(map(sigmoid,out_node)) 
        
        # de-Normalize to output
        X_Velocity = (out_node[0]*(max_output1 - min_output1)) + min_output1
        Y_Velocity = (out_node[1]*(max_output2 - min_output2)) + min_output2
                
        print([Y_Velocity, X_Velocity])
        
        return [X_Velocity, Y_Velocity]

