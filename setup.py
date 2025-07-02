from setuptools import setup

setup(
    package_data={
        "contest_helper.cli": ["templates/*", "templates/*/*"],
    },
)
