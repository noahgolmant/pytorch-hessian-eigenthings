"""setup.py for hessian_eigenthings"""

from setuptools import setup, find_packages

install_requires = [
    'numpy>=0.14',
    'torch>=0.4',
    'scipy>=1.2.1'
]

setup(name="hessian_eigenthings",
      author="Noah Golmant",
      install_requires=install_requires,
      packages=find_packages(),
      description='Eigendecomposition of model Hessians in PyTorch!',
      version='0.0.1')
