from plangid.dataset import Dataset
from plangid.pipeline import LanguagePipeline


def test_fit_predict():
    dataset = Dataset()
    df = dataset.to_df()
    df = df.head(n=100)
    lp = LanguagePipeline()
    lp.fit(df)
    lp.predict(df)
