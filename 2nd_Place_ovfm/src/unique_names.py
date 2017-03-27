import glob
import pprint

files = glob.glob('../example_extracted/*/structures.dat')
unique_names = set()
for filename in files:
    with open(filename, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.split('|')]
            unique_names.update(parts)
pprint.pprint(unique_names)
