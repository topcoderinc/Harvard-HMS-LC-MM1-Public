import os
import glob
import cPickle as pickle
PARTS = 12

def pack_jpgs(d):
    l = glob.glob('heatmaps/*.jpg')
    names = list(set(map(lambda x: x[:x.rfind('_')], l)))
    assert(len(names) == d)
    for name in names:
        package = []
        for x in range(PARTS):
            package.append(bytearray(open(name+'_'+str(x)+'.jpg', 'rb').read()))
        f = open(name+'.p', 'w')
        pickle.dump(package, f)
        f.close()
    os.system('rm heatmaps/*.jpg')

l = open('scans_all.csv', 'r').readlines()
os.system('rm -f masks.csv')

while len(l) > 0:
    print len(l)
    t = l[:10]
    f = open('scans.csv', 'w')
    for i in t:
      print >>f, i,
    f.close()
    os.system('./vis --path example_extracted --heat 200')
    pack_jpgs(len(t))
    l = l[10:]
    os.system('cat .masks >> masks.csv')
