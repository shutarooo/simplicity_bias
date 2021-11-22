import json
import matplotlib.pyplot as plt
from lz_complexity import lz_compression

with open('data/func_freq/unit_uniform.json') as f:
    unit_data = json.load(f)  # ファイルオブジェクトfinをデコードする

with open('data/func_freq/wide_uniform.json') as f:
    wide_data = json.load(f)  # ファイルオブジェクトfinをデコードする

unit = [data[1]/(10**6) for data in unit_data]
#wide = [data[1]/(10**6) for data in wide_data]
x_unit = [data[0] for data in unit_data]
#x_wide = [data[0] for data in wide_data]
for i, data in enumerate(x_unit):
    data_char = ''
    for d in data:
        data_char += str(bin(int(d)))
    if not ('0' in data_char) or not ('1' in data_char):
        x_unit[i] = 7
    else:
        x_unit[i] = 7*(lz_compression(list(data)) + lz_compression(list(data)[::-1]))/2
print(x_unit)

plt.scatter(x_unit, unit)
plt.show()







