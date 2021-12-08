import json
import matplotlib.pyplot as plt

with open('learning/model_10_input/info_9.json') as f:
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

Figure = plt.figure() #全体のグラフを作成
ax1 = Figure.add_subplot(2,2,1) #1つ目のAxを作成
ax2 = Figure.add_subplot(2,2,2) #2つ目のAxを作成
ax3 = Figure.add_subplot(2,2,3) #2つ目のAxを作成

ax1.plot(epoch, comp)
ax1.set_xlabel('epoch')
ax1.set_ylabel('lz_complexity')

ax2.plot(epoch, norm[:-1])
ax2.set_xlabel('epoch')
ax2.set_ylabel('norm')


ax3.plot(epoch, diff)
ax3.set_xlabel('epoch')
ax3.set_ylabel('diff')


plt.show()
plt.savefig('images/comp_process/model_10_input/model_8.pdf')

'''
plt.plot(epoch, comp)
plt.xlabel('epoch')
plt.ylabel('lz_complexity')
plt.title('learning process')
plt.savefig('images/comp_process/model_10_input/model_5.pdf')
plt.show()
'''
