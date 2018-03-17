# -*- coding: utf-8 -*-

from distutils.core import setup
import shutil
shutil.copy('README.md', 'pyDelves/README.md')

setup(name='pyDelves',
      version='2.5',
      description='Finds roots of an analytical function',
      author="Peter Bingham",
      author_email="petersbingham@hotmail.co.uk",
      packages=['pydelves'],
      package_data={'pydelves': ['tests/*', 'README.md']}
     )
