def lz_compression(S):   
    # uがprefixの長さで、vが新たにprefixに追加する部分の長さ。u+vがlength nになるまで繰り返す。
    i = 0        # i = p - 1, p is the pointer (see above)
    C = 1
    u = 1      # the length of the current prefix
    v = 1      # the length of the current component for the current pointer p
    vmax = v   # the final length used for the current component (largest on all the possible pointers p)
    n = len(S)
    while u + v <= n:              # sequenceの長さを超えない限り繰り返し
        if S[i + v - 1] == S[u + v - 1]:      # prefixとcomponentが一致する最大の長さをvとして求める
            v = v + 1
        else:
            vmax = max(v, vmax)      # 最も長いcomponentを更新
            i = i + 1                        # pointerの位置を更新
            if i == u:                      # all the pointers have been treated
                C = C + 1
                u = u + vmax
                v = 1
                i = 0
                vmax = v
            else:
                v = 1
    if v != 1:
        C = C+1
    return C