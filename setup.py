from setuptools import setup

setup(
  name='MOM_analysis',
  version='0.1.0',
  author='Chris Bladwell',
  author_email='c.bladwell@unsw.edu.au',
  packages=['hat_average'],
  scripts=['hat_average/mf_figures'],
  url='',
  license='',
  description='tools to analyse MOM code for time averaging and tracer analysis',
  long_description=open('README.txt').read(),
  install_requires=[
      "xarray",
      "numpy",
  ],
)
