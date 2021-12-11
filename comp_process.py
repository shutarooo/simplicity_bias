import json
import matplotlib.pyplot as plt

with open('learning/model_flat/info_1.json') as f:
    data = json.load(f)  # ファイルオブジェクトfinをデコードする

print(data['epochs'])

epoch_comp = data['comp_process']
epoch = []
comp = []
for d in epoch_comp:
    epoch.append(d[0])
    comp.append(d[1])
norm = data['norm']
diff = data['diff']
loss = data['loss_process']
flatness = data['flatness']
epoch_50 = []

for i in range(len(flatness)):
    epoch_50.append(50*(i+1))

Figure = plt.figure()  # 全体のグラフを作成
ax1 = Figure.add_subplot(2, 3, 1)  # 1つ目のAxを作成
ax2 = Figure.add_subplot(2, 3, 2)  # 2つ目のAxを作成
ax3 = Figure.add_subplot(2, 3, 3)  # 2つ目のAxを作成
ax4 = Figure.add_subplot(2, 3, 4)  # 2つ目のAxを作成
ax5 = Figure.add_subplot(2, 3, 5)  # 2つ目のAxを作成

ax1.plot(epoch, comp)
ax1.set_xlabel('epoch')
ax1.set_ylabel('lz_complexity')

ax2.plot(epoch, norm[:-1])
ax2.set_xlabel('epoch')
ax2.set_ylabel('norm')

ax3.plot(epoch, diff)
ax3.set_xlabel('epoch')
ax3.set_ylabel('diff')

ax4.plot(epoch, loss[:-1])
ax4.set_xlabel('epoch')
ax4.set_ylabel('loss')

ax5.plot(epoch_50, flatness)
ax5.set_xlabel('epoch')
ax5.set_ylabel('flatness')


plt.show()
plt.savefig('images/comp_process/model_flat/model_1.pdf')

'''
plt.plot(epoch, comp)
plt.xlabel('epoch')
plt.ylabel('lz_complexity')
plt.title('learning process')
plt.savefig('images/comp_process/model_10_input/model_5.pdf')
plt.show()
'''
