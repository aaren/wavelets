try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
        'description': 'Continuous wavelet analysis in Python',
        'author': "Aaron O'Leary",
        'url': 'http://github.com/aaren/wavelets',
        'author_email': 'eeaol@leeds.ac.uk',
        'version': '0.1',
        'packages': ['wavelets'],
        'name': 'wavelets',
        }

setup(**config)
