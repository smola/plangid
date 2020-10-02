import nox


@nox.session(reuse_venv=True)
def lint_black(session):
    session.install("black")
    session.run("black", "--check", "plangid", "noxfile.py")


@nox.session(reuse_venv=True)
def lint_flake8(session):
    session.install("flake8")
    session.run("flake8", "plangid")


@nox.session(python=["3.6", "3.7", "3.8"])
def test(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run("pytest", "--ignore=language-dataset")


@nox.session(python="3.8")
def dist(session):
    session.install("poetry")
    session.run("poetry", "build")
