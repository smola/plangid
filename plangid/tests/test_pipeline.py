import pandas as pd
import pytest

from plangid.dataset import Dataset
from plangid.pipeline import LanguagePipeline


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return Dataset()


@pytest.fixture(scope="module")
def df(dataset: Dataset) -> pd.DataFrame:
    return dataset.to_df()


@pytest.fixture(scope="function")
def lp() -> LanguagePipeline:
    return LanguagePipeline()


@pytest.fixture(scope="module")
def fitted_lp(df: pd.DataFrame) -> LanguagePipeline:
    return LanguagePipeline().fit(df.head(n=100))


def test_predict(fitted_lp: LanguagePipeline, df: pd.DataFrame) -> None:
    assert fitted_lp.predict(df.head(n=100)) is not None


def test_predict_proba(fitted_lp: LanguagePipeline, df: pd.DataFrame) -> None:
    assert fitted_lp.predict_proba(df.head(n=100)) is not None


def test_fit_save_load(tmp_path, fitted_lp, df):
    model_path = tmp_path / "model"
    fitted_lp.save(model_path)
    new_lp = LanguagePipeline.load(model_path)
    assert isinstance(new_lp, LanguagePipeline)
    assert new_lp.predict(df.head(n=100)) is not None
