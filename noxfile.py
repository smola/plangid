import nox


python_versions = ["3.7", "3.8"]
default_python = "3.8"


@nox.session(python=default_python, reuse_venv=True)
def lint_black(session):
    session.install("black")
    session.run("black", "--check", "plangid", "noxfile.py")


@nox.session(python=default_python, reuse_venv=True)
def lint_flake8(session):
    session.install("flake8")
    session.run("flake8", "plangid")


@nox.session(python=python_versions)
def test(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run("pytest", "--ignore=language-dataset")


@nox.session(python=default_python)
def dist(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run(
        "python", "-m", "plangid.cli", "train", "--model-path=plangid/model.pickle.gz"
    )
    session.run("poetry", "build")
