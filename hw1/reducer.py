#!/usr/bin/python3
import sys

cj, mj, vj = 0, 0, 0

for line in sys.stdin:
    try:
        ck, mk, vk = line.strip().split('\t')

        ck = int(ck)
        mk = float(mk)
        vk = float(vk)
    except:
        continue

    v = (cj * vj + ck * vk) / (cj + ck) + cj * ck * ((mj - mk) / (cj + ck)) ** 2
    cj += ck

    mj, vj = mk, v

print(cj, mj, vj, sep='\t')