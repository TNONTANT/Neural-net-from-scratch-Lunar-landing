import util
import pandas as pd


# input data
df = pd.read_csv("ce889_dataCollection.csv",header=None)

# preprocess data
processed = util.pre_processing(df)

###############################Find best network number(bn)######################## 

bn = util.Node(10,50)
bestnode = bn[0]
bn_rms_train = bn[1]
bn_rms_test = bn[2]

m_bn = min(bn_rms_test)
bn_index = bn_rms_test.index(m_bn)
FitNode = bestnode[bn_index]

################################Find best Learning Rate#############################

Lr = util.LearningRate(10, FitNode)
LR = Lr[2]
lr_rms_train = Lr[0]
lr_rms_test = Lr[1]

m_lr = min(lr_rms_test)
lr_index = lr_rms_test.index(m_lr)
FitLr = LR[lr_index]

################################Find fit Momentum##################################

Mt = util.Momentum(10, FitNode, FitLr)
MT = Mt[2]
mt_rms_train = Mt[0]
mt_rms_test = Mt[1]

m_mt = min(mt_rms_test)
mt_index = mt_rms_test.index(m_mt)
FitMt = MT[mt_index]

################################Train and Test#####################################

# Training
Training = util.NeuralNetwork(200, FitLr, FitMt, FitNode, processed[0], processed[1], 
                            processed[3], processed[4])

train_rms_per_ep = Training[0]
train_each_y_hat = Training[1]
train_w_hid_ep = Training[2]
train_w_out_ep = Training[3]

# Testing
Testing = util.TestNeural(200, FitLr, FitMt, FitNode, processed[5], processed[6], 
                        processed[7], processed[8],
                        train_w_hid_ep, train_w_out_ep, False)

test_rms_per_ep = Testing

m_test = min(test_rms_per_ep)
test_index = test_rms_per_ep.index(m_test)

# weight for rocket
rocket_w_hid = train_w_hid_ep[test_index]
rocket_w_out = train_w_out_ep[test_index]

print("best Network num =", FitNode, "\n")
print("best Learning Rate =", FitLr, "\n")
print("best Momentum =", FitMt, "\n")

############# use these weight into NeuralNetHolder.py file ##################
print("Weight to hiddenLayer for Neural NetHolder is:\n",rocket_w_hid, "\n")
print("Weight to outputLayer for Neural NetHolder is:\n",rocket_w_out, "\n")

# use pandas to save weight in both hidden Layer and Output Layer
w_hid = pd.DataFrame(rocket_w_hid)
w_out = pd.DataFrame(rocket_w_out)

PATH = "/Users/nontanatto/Desktop/Neural-net-from-scratch-Lunar-landing/"
w_hid.to_csv(PATH + "w_hid.csv", index=False )
w_hid.to_csv(PATH + "w_out.csv", index=False )


###### plot graph #######

# importing libraries
import matplotlib.pyplot as plt

    
figure, axis = plt.subplots(2, 2)
    
# For Network
axis[0, 0].plot(bn_rms_train)
axis[0, 0].plot(bn_rms_test)
axis[0, 0].set_title("Network")
    
# For Learning Rate
axis[0, 1].plot(lr_rms_train)
axis[0, 1].plot(lr_rms_test)
axis[0, 1].set_title("Learning Rate")
    
# For Momentum
axis[1, 0].plot(mt_rms_train)
axis[1, 0].plot(mt_rms_test)
axis[1, 0].set_title("Momentum")
    
# For Train and Test Model
axis[1, 1].plot(train_rms_per_ep)
axis[1, 1].plot(test_rms_per_ep)
axis[1, 1].set_title("Train and Test Model")

# Combine all the operations and display
plt.show()

print("Final Model, data ce889_dataCollection.csv \n, 10 epoch of find each parameter \n,train 70% test 30%")