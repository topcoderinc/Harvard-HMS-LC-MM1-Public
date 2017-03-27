import argparse
import os

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to data')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    print args
    os.system('ls ' + args.path + ' > .tmp')
    patients = map(lambda s: s[:-1], open('.tmp').readlines())
    print patients
    f = open('scans.csv', 'w')
    for p in patients:
        os.system('ls ' + '/'.join([args.path, p, 'auxiliary']) + ' > .tmp')
        scans = sorted(map(int, map(lambda s: s[:-5], open('.tmp').readlines())))
        print 'patient', p, 'scans', scans
        for scan in scans:
            lines = open('/'.join([args.path, p, 'auxiliary', str(scan)+'.dat'])).readlines()
            x0 = None
            y0 = None
            dx = None
            dy = None
            slice_thickness = None
            for line in lines:
                v = line[:-1].split(',')
                tag = v[0]
                if tag == '(0018.0050)':
                    slice_thickness = v[1]
                elif tag == '(0020.0032)':
                    x0 = v[1]
                    y0 = v[2]
                elif tag == '(0028.0030)':
                    dx = v[1]
                    dy = v[2]
            assert(x0 and y0 and dx and dy)
            print >>f, ','.join([p, str(scan), x0, y0, dx, dy, slice_thickness])

        #print open('/'.join([args.path, p, ])).readlines()
    f.close()
