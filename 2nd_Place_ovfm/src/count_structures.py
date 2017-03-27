import json
import sys
from collections import Counter

c = Counter()

count = 0
with open(sys.argv[1], 'r') as f:
    for line in f:
        count += 1
        data = json.loads(line)
        c = c + Counter(list(data['structures'].keys()))

for k, v in c.items():
    ratio = v / float(count)
    print(k, v, ratio, 1.0 / ratio)
