from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.externals import joblib
#import pandas as pd
from extract_features import contour_extractors

nb_extractors = len (contour_extractors)

def load_data(features_dataset):

    with open(features_dataset) as f:
        train = [[] for _ in range(nb_extractors)]
        target_precision = [[] for _ in range(nb_extractors)]
        target_recall = [[] for _ in range(nb_extractors)]
        for row in f.readlines():
            scan_id, precision, recall, extractor_id, *features = row.strip().split(' ')
            extractor_id = int(extractor_id)
            train[extractor_id].append([float(x) for x in features])
            target_precision[extractor_id].append(float(precision))
            target_recall[extractor_id].append(float(recall))

    return train, target_precision, target_recall


def learn(train, target):

    clf = RandomForestRegressor(n_estimators=100, n_jobs=40)
    #clf = svm.SVR(gamma=0.0001, C=100)
    clf.fit(train, target)
    return clf

def predict(clf, valid, target):

    #FIXME
    #clf = joblib.load(modelpath)
    pred = clf.predict(valid)

    print(abs(target - pred).sum())


if __name__ == '__main__':

    data = "/home/gerey/hms_lung/data/example_extracted/features6.ssv"
    model_precision = "/home/gerey/hms_lung/data/example_extracted/precision_randomforest6-%s.clf"
    model_recall = "/home/gerey/hms_lung/data/example_extracted/recall_randomforest6-%s.clf"

    train, target_precision, target_recall = load_data(data)

    for target, output in [(target_precision, model_precision),(target_recall, model_recall)]:
        for i in range(len(contour_extractors)):
            clf = learn(train[i], target[i])
            joblib.dump(clf, output % (i,))

