from plangid.dataset import Dataset


def test_samples():
    dataset = Dataset()
    samples = list(dataset.samples())
    assert len(samples) > 0
    assert all([len(s) == 2 for s in samples])


def test_to_df():
    dataset = Dataset()
    df = dataset.to_df()
    assert set(["class_name", "class_index", "content", "path", "filename"]) == set(
        df.columns
    )
    assert len(df["class_index"].unique()) - 1 == df["class_index"].max()
