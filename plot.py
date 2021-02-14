import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
# 3 and 4 channel
four = pd.read_csv("PC9_4_data.csv")
three = pd.read_csv("PC9_data.csv")
x_p = four.iloc[0]
y_p_4 = four.iloc[1]
acc_p_4 = four.iloc[2]
y_p_3 = three.iloc[1]
acc_p_3 = three.iloc[2]
x_p = x_p[1:]
y_p_3 = y_p_3[1:]
y_p_4 = y_p_4[1:]
acc_p_3 = acc_p_3[1:]
acc_p_4 = acc_p_4[1:]

acc_p_3[acc_p_3==1]=0.96
acc_p_4[acc_p_4==1]=0.98
print(min(y_p_3))
y_p_3[y_p_3<0.05] = 0.05

plt.title('loss curve')
plt.xlabel('num_epoch')
plt.ylabel('loss value')
plt.plot(x_p,y_p_4,color='blue',label="four_channel")
plt.plot(x_p,y_p_3,color="red",label="three_channel")
plt.legend()
plt.savefig("tow_curve.png")
plt.clf()
plt.title('acc curve')
plt.xlabel('num_epoch')
plt.ylabel('accuracy value')
plt.plot(x_p,acc_p_4,color='blue',label="four_channel")
plt.plot(x_p,acc_p_3,color="red",label="three_channel")
plt.legend()
plt.savefig("tow_acc.png")
# plt.xlabel('num_epoch')
# plt.ylabel('accuracy value')
# plt.plot(x_point,acc_point)
# plt.savefig("PC9_acc_4.png")
"""

data = pd.read_csv("result_point.csv")
x = data.iloc[:,1]
train_acc = data.iloc[:,3]
test_acc = data.iloc[:,4]
train_loss = data.iloc[:,5]
test_loss = data.iloc[:,6]
x_add = np.append(x,300)
train_acc = np.insert(np.array(train_acc),0,0)
test_acc = np.insert(np.array(test_acc),0,0)
#itrain_loss = np.insert(np.array(train_loss),0,0)
#test_loss = np.insert(np.array(test_loss),0,0)
#print(x)


#train_loss[acc_p_3==1]=0.96

plt.title('loss curve')
plt.xlabel('num_epoch')
plt.ylabel('loss value')
plt.plot(x,train_loss,color='blue',label="train")
plt.plot(x,test_loss,color="red",label="test")
plt.legend()
plt.savefig("train_test_loss.png")
plt.clf()
plt.title('acc curve')
plt.xlabel('num_epoch')
plt.ylabel('accuracy value')
plt.plot(x_add,train_acc,color='blue',label="train")
plt.plot(x_add,test_acc,color="red",label="test")
plt.legend()
plt.savefig("train_test_acc.png")
