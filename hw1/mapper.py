#!/usr/bin/env python3
import sys
import csv

count = 0
cum_sum = 0
sqr_cum_sum = 0

for row in csv.reader(sys.stdin, delimiter=","):

    try:
        price = int(row[-7])
    except:
        continue

    cum_sum += price
    sqr_cum_sum += price ** 2
    count += 1

mean = cum_sum / count
var = sqr_cum_sum / count - mean ** 2

print(count, mean, var, sep='\t')