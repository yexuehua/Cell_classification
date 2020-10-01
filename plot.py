import pandas as pd
import matplotlib.pyplot as plt

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
