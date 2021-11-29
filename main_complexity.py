import json
import matplotlib.pyplot as plt
from lz_complexity import lz_compression

with open('data/func_freq/NN/unit_uniform.json') as f:
    unit_data = json.load(f)  # ファイルオブジェクトfinをデコードする

x_out = []
y_out = []
for i, data in enumerate(unit_data):
    if data[1] == 1:
        continue
    
    # [10進, 10進, 10進]となっているlistをbinaryの文字列に直す。
    data_char = ''
    for j in range(len(data[0])):
        d_bin = str(bin(int(data[0][j])))[2:]
        if (j == 0 or j == 1) and len(d_bin) < 60:
            d_bin = '0'*(60-len(d_bin)) + d_bin
        elif j==2 and len(d_bin) < 8:
            d_bin = '0'*(8-len(d_bin)) + d_bin
        data_char += d_bin
    
    y = 0
    if (not '0' in data_char) or (not '1' in data_char):
        #x_unit[i] = 7
        x_out.append(7)
    else:
        y = 7*(lz_compression(list(data_char)) + lz_compression(list(data_char[::-1])))/2
        #x_unit[i] = 7*(lz_compression(list(data)) + lz_compression(list(data)[::-1]))/2
        x_out.append(y)
        if 30 < y <=33:
            print(y, data_char)
    y_out.append(data[1]/(10**6))
    '''
    if i < 30:
        print(data_char)
    '''
#print(x_unit)

plt.scatter(x_out, y_out)
plt.yscale('log')
plt.xlabel('lz_complexity')
plt.ylabel('probability')

plt.title('NN_unit')
#plt.savefig('images/complexity_bias_2/NN_unit.pdf')
plt.show()










