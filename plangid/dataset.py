import os
import os.path
import numpy as np
import pandas as pd
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

MAX_SIZE = 10 * 1024


class Dataset:
    def __init__(self, path=None, min_samples=None):
        if not path:
            path = os.environ["LANGUAGE_DATASET_PATH"]
        self.path = path
        self.min_samples = min_samples

    @staticmethod
    def load_sample(path):
        filename = os.path.basename(path)
        content = Dataset._read_content(path)
        return {"filename": filename, "content": content}

    def samples(self):
        dataset_path = os.path.join(self.path, "dataset.yml")
        with open(dataset_path, "r") as f:
            dataset = yaml.load(f, Loader=Loader)
        return ((k, v["annotations"]["vote"]) for k, v in dataset["files"].items())

    def to_df(self):
        class_name = []
        filename = []
        content = []
        path = []
        for p, c in self.samples():
            if c == "Unknown":
                continue
            class_name.append(c)
            path.append(p)
            s = Dataset.load_sample(os.path.join(self.path, p))
            filename.append(s["filename"])
            content.append(s["content"])
        df = pd.DataFrame()
        df["class_name"] = pd.Series(class_name)
        df["filename"] = pd.Series(filename)
        df["content"] = pd.Series(content)
        df["path"] = pd.Series(path)

        classes = df.groupby(["class_name"]).count().reset_index()
        if self.min_samples:
            classes = classes.loc[classes["path"] >= self.min_samples]
        classes = classes.reset_index(drop=True).reset_index(drop=False)
        classes = classes.rename(columns={"index": "class_index"})
        classes["class_name"] = classes["class_name"].astype(np.str)
        classes = classes[["class_name", "class_index"]]

        df = df.merge(classes, on=["class_name"], how="inner")

        return df

    @staticmethod
    def _read_content(file_path):
        with open(file_path, "rb") as f:
            return f.read()[:MAX_SIZE]
