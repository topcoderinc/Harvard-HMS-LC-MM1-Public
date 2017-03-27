from unittest import TestCase

import itertools

from helpers.input_data import Dataset, stack

sample_path = "/home/gerey/hms_lung/data/example_extracted_sample"
nb6 = 199-2  # First and last slice removed
nb2 = 124-2

class TestDataset(TestCase):

    def test_features_and_binary_target(self):

        #self.assertEqual(scan[200], 'ANON_LUNG_TC006')
        #scan1 = [0]*144
        #scan1[62:73] = [1]*11
        scan6 = [0]*nb6
        scan6[93-1:99-1] = [1]*6
        scan2 = [0]*nb2
        scan2[65-1:77-1] = [1]*12

        ds = Dataset(sample_path)
        tmp = [ [a,b,d] for a,b,c,d in itertools.islice(ds.features_and_binary_targets(), nb6+nb2) ]
        scan, slice, res = zip(*tmp)
        # sanity checks for the rest of the tests.
        # but order shouldn't matter.
        # todo: more flexible
        self.assertEqual(scan[0], 'ANON_LUNG_TC006')
        self.assertEqual(scan[nb6], 'ANON_LUNG_TC002')
        self.assertEqual(slice[nb6], 2)
        self.assertEqual(slice[nb6+1], 3)
        self.assertSequenceEqual(res[:nb6], scan6)
        self.assertSequenceEqual(res[nb6:], scan2)


    def test_batch_of_binary(self):
        ds = Dataset(sample_path)
        _, targets= stack([c,d] for a,b,c,d in itertools.islice(ds.features_and_binary_targets(), 4))
        self.assertEqual(targets.shape, (4,))


    def test_targets(self):
        ds = Dataset(sample_path)
        tmp = [ [a,b,e] for a,b,c,d,e in itertools.islice(ds.images_and_targets(resize=False), nb6) ]
        scans, slices, targets = zip(*tmp)
        sums = [t.sum() for t in targets]
        self.assertEqual(scans[0], 'ANON_LUNG_TC006')
        self.assertEqual(sums[:93],   [0]*93)
        self.assertEqual(sums[93:99], [50., 123., 195., 192., 147., 41.])
        self.assertEqual(sums[99:],   [0]*(nb6-99))


    def test_filter_out_blank(self):
        ds = Dataset(sample_path)
        tmp = [ [a,b,d] for a,b,c,d in itertools.islice(ds.features_and_targets(resize=False, filterBlank=True), 7) ]
        scans, slices, targets = zip(*tmp)
        sums = [t.sum() for t in targets]
        self.assertEqual(scans[0], 'ANON_LUNG_TC006')
        self.assertEqual(scans[6], 'ANON_LUNG_TC002')
        self.assertEqual(sums[:6], [50., 123., 195., 192., 147., 41.])
        self.assertEqual(sums[6], 320.)




