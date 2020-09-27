from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

from .dataset import Dataset
from .pipeline import LanguagePipeline


def main():
    CV_SPLIT = 10

    dataset = Dataset(min_samples=CV_SPLIT)
    df = dataset.to_df()

    lp = LanguagePipeline()

    preds_proba = cross_val_predict(
        lp, df, df["class_index"], method="predict_proba", cv=CV_SPLIT
    )
    preds = preds_proba.argmax(axis=1)

    accuracy = accuracy_score(df["class_index"], preds)
    print("Accuracy: %f" % accuracy)

    print("Errors:")

    import pandas as pd

    class_names = df.sort_values(by=["class_index"], ascending=True)[
        "class_name"
    ].unique()
    df_preds = pd.DataFrame(data=preds_proba, columns=class_names)
    df_eval = pd.concat([df, df_preds], axis=1, keys=["meta", "preds"])
    df_eval = df_eval.loc[
        df_eval["preds"].idxmax(axis=1) != df_eval["meta"]["class_name"]
    ]
    df_eval["max_prob"] = df_eval["preds"].max(axis=1)
    df_eval = df_eval.sort_values(by=["max_prob"], ascending=False)
    for _, row in df_eval.iterrows():
        path = row["meta"]["path"]
        class_name = row["meta"]["class_name"]
        k = 3
        top_k = row["preds"].sort_values(ascending=False)[:k]
        fields = [path, class_name]
        for cn, v in zip(top_k.index, top_k.values):
            fields.append("%s (%.4f)" % (cn, v))
        print("\t" + "\t".join(fields))
