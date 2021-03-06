import json
import matplotlib.pyplot as plt

with open('data/freq_test/normalized_no_bias.json') as f:
    unit_data = json.load(f)  # ファイルオブジェクトfinをデコードする

'''
with open('data/func_freq/wide_uniform.json') as f:
    wide_data = json.load(f)  # ファイルオブジェクトfinをデコードする
'''


unit = [data[1]/(10**6) for data in unit_data]
#wide = [data[1]/(10**6) for data in wide_data]
x_unit = [i+1 for i in range(len(unit_data))]
#x_wide = [i+1 for i in range(len(wide_data))]

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
#ax2 = fig.add_subplot(1, 2, 2)

ax1.plot(x_unit, unit)
#ax1 = plt.gca()
ax1.set_yscale('log')
ax1.set_xscale('log')

plt.xlabel('rank')
plt.ylabel('prob')

plt.title('no bias')

#ax2.plot(x_wide, wide)
#ax2 = plt.gca()
#ax2.set_yscale('log')

#plt.savefig('images/normalized/norm_9.pdf')
plt.show()
