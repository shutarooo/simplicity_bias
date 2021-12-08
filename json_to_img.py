import json
import matplotlib.pyplot as plt

with open('data/func_freq_norm/comp_freq/norm_9.json') as f:
    unit_data = json.load(f)  # ファイルオブジェクトfinをデコードする

prob = []
comp = []

all_comp = []

comp_freq_map = {}

for data in unit_data:
    all_comp.append(data[0])

    if data[0] in comp_freq_map:
        
        comp_freq_map[data[0]] += data[1]
        
    else:
        comp_freq_map[data[0]] = data[1]

    if data[1] == 1:
        continue
    comp.append(data[0])
    prob.append(data[1]/10**6)

'''
plt.scatter(comp, prob)
plt.yscale('log')
plt.xlabel('lz_complexity')
plt.ylabel('probability')
plt.title('normalized_4.5')
plt.savefig('images/normalized/comp_prob/norm_45.pdf')
plt.show()

plt.hist(all_comp)
plt.yscale('log')
plt.xlabel('lz_complexity')
plt.ylabel('number of different functions appeared')
plt.title('normalized_4.5')
plt.savefig('images/normalized/all_func/norm_45.pdf')
plt.show()
'''

comp_freq_map = sorted(comp_freq_map.items())
comp = []
freq = []
for data in comp_freq_map:
    comp.append(data[0])
    freq.append(data[1])

#plt.plot(list(comp_freq_map.keys()), list(comp_freq_map.values()))
plt.plot(comp, freq)
plt.yscale('log')
plt.xlabel('lz_complexity')
plt.ylabel('frequency')
plt.title('normalized_')
plt.savefig('images/normalized/comp_prob_2/norm_9.pdf')
plt.show()
