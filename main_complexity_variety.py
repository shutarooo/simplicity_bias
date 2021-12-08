import json
import matplotlib.pyplot as plt
from lz_compression import lz_compression

with open('data/func_freq_norm/raw/norm_135.json') as f:
    unit_data = json.load(f)  # ファイルオブジェクトfinをデコードする

complexity_list = []
freq_list = []
comp_freq_list = []
cnt=0
for i, data in enumerate(unit_data):
    freq_list.append(data[1])
    # [10進, 10進, 10進]となっているlistをbinaryの文字列に直す。
    data_char = ''
    for j in range(len(data[0])):
        d_bin = str(bin(int(data[0][j])))[2:]
        if (j == 0 or j == 1) and len(d_bin) < 60:
            d_bin = '0'*(60-len(d_bin)) + d_bin
        elif j==2 and len(d_bin) < 8:
            d_bin = '0'*(8-len(d_bin)) + d_bin
        data_char += d_bin
    
    
    if (not '0' in data_char) or (not '1' in data_char):
        #x_unit[i] = 7
        comp_freq_list.append([7, data[1]])
    else:
        y = 7*(lz_compression(list(data_char)) + lz_compression(list(data_char[::-1])))/2
        #x_unit[i] = 7*(lz_compression(list(data)) + lz_compression(list(data)[::-1]))/2
        comp_freq_list.append([y, data[1]])
    if cnt % 10**5 ==0:
        print(f'{cnt/10**4}%, finished.')
    cnt += 1

path_file = './data/func_freq_norm/comp_freq/norm_135.json'
with open(path_file, 'w') as f:
        json.dump(comp_freq_list , f)

'''
plt.hist(func_complexity_list)
plt.yscale('log')
plt.xlabel('lz_complexity')
plt.ylabel('number of different functions')

plt.title('DNN wide')
plt.savefig('images/complexity_variety/DNN_unit.pdf')
plt.show()
'''