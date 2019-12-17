from plangid.dataset import Dataset
from plangid.pipeline import LanguagePipeline


def test_fit():
    dataset = Dataset()
    df = dataset.to_df()
    df = df.head(n=100)
    lp = LanguagePipeline()
    lp.pipeline.fit(df, df["class_index"])
