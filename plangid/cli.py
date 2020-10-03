import click
import os.path

from plangid.dataset import Dataset
from plangid.pipeline import LanguagePipeline


_HOME = os.path.join(os.path.expanduser("~"), ".plangid")
_DEFAULT_MODEL_NAME = "model.pickle.gz"
_DEFAULT_MODEL_PATH = os.path.join(_HOME, _DEFAULT_MODEL_NAME)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--model-path",
    envvar="MODEL_PATH",
    show_envvar=True,
    show_default=True,
    type=str,
    default=_DEFAULT_MODEL_PATH,
)
@click.option("--explain", is_flag=True)
@click.argument("file", type=click.Path(exists=True, file_okay=True))
def detect(model_path, explain, file):
    if model_path is _DEFAULT_MODEL_PATH:
        if not os.path.exists(model_path):
            model_path = None
    lp = LanguagePipeline.load(path=model_path)
    if explain:
        print(lp.explain(file))
    else:
        print(lp.predict(file)[0])


@cli.command()
@click.option(
    "--dataset-path",
    envvar="LANGUAGE_DATASET_PATH",
    show_envvar=True,
    is_flag=False,
    multiple=False,
    default="language-dataset/data",
)
@click.option(
    "--model-path",
    envvar="MODEL_PATH",
    show_envvar=True,
    show_default=True,
    type=str,
    default=_DEFAULT_MODEL_PATH,
)
def train(dataset_path, model_path):
    dataset = Dataset(path=dataset_path)
    lp = LanguagePipeline()
    lp.fit(dataset)
    lp.save(path=model_path)


if __name__ == "__main__":
    cli()
