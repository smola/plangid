from hyperopt.pyll import scope
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np

from .dataset import Dataset
from .pipeline import LanguagePipeline

CV_SPLIT = 5

dataset = Dataset(min_samples=CV_SPLIT)
df = dataset.to_df()


@scope.define
def train(
    min_ngram=3,
    max_ngram=5,
    min_df=1,
    max_df=1.0,
    n_estimators=100,
    min_samples_split=2,
):
    lp = LanguagePipeline(
        ngram_range=(int(min_ngram), int(max_ngram)),
        min_df=int(min_df),
        max_df=max_df,
        n_estimators=int(n_estimators),
        min_samples_split=int(min_samples_split),
    )

    feature_extraction = lp.pipeline.named_steps["feature_extraction"]
    clf = lp.pipeline.named_steps["clf"]

    data = feature_extraction.fit_transform(df)

    from sklearn.model_selection import cross_val_predict

    preds = cross_val_predict(clf, data, df["class_index"], cv=CV_SPLIT)

    # TODO: try AUC ROC (ovr, weighted)
    from sklearn.metrics import accuracy_score

    score = accuracy_score(df["class_index"], preds)

    return 1.0 - score


def main():

    from hyperopt import hp

    space = scope.train(
        min_ngram=hp.quniform("min_ngram", 1, 2, 1),
        max_ngram=hp.quniform("max_ngram", 2, 6, 1),
        min_df=hp.quniform("min_df", 2, 6, 1),
        max_df=hp.uniform("max_df", 0.95, 1.0),
        n_estimators=hp.quniform("n_estimators", 100, 500, 1),
        min_samples_split=hp.quniform("min_samples_split", 2, 6, 1),
    )

    trials = Trials()
    best = fmin(train, space=space, algo=tpe.suggest, max_evals=100, trials=trials,)
