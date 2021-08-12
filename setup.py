from setuptools import setup

setup(
  name='MOM_analysis',
  version='0.1.0',
  author='Chris Bladwell',
  author_email='c.bladwell@unsw.edu.au',
  packages=['hat_average'],
  scripts=['bin/script1','bin/script2'],
  url='',
  license='',
  description='tools to analyse MOM code for time averaging and tracer analysis',
  long_description=open('README.txt').read(),
  install_requires=[
      "xarray",
      "numpy",
  ],
)
