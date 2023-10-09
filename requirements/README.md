# Project Requirements

This folder contains all requirements files for the project. The base requirements are located in the `base.txt` file.
Files prefixed with `_` are only meant for development and testing purposes. In general, each subdomain of the project
have a `<domain>.txt` file that contains the necessary requirements for using that subdomain and a `<domain>_test.txt`
file that contains the necessary requirements for testing that subdomain.

To install all extra requirements such that all tests can be run, use the following command:

```bash
pip install -r requirements/_devel.txt  # unittests
pip install -r requiremnets/_integrate.txt  # integration tests

```

To install all extra requirements such that the documentation can be build, use the following command:

```bash
pip install -r requirements/_docs.txt
```

## CI/CD upper bounds

For Ci stability, we have set for all package versions upper bounds (the latest version), so with any sudden release,
we won't put our development on fire. The continues updated of these upper bounds are managed by dependabot.
Note that these upper bounds are lifters when installing a package from the source or as a package.
If you want to preserve/enforce restrictions on the latest compatible version, add "strict" as an in-line comment.
