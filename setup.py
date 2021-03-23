from setuptools import setup

setup(name='evalSemanticSeg',
      version='0.0.1',
      description='Evaluate semantic segmentation for cutomized data, for models trained on cityscapes',
      url='https://github.com/ddatta-DAC/evalSemanticSeg',
      author='Debanjan Datta',
      author_email='ddatta@cs.vt.com',
      license='MIT',
      packages=['evalSemanticSeg'],
      install_requires=[
          'numpy',
      ],
      zip_safe=False)
