from setuptools import setup

setup(name='evalSemanticSeg',
      version='0.0.5',
      description='Evaluate semantic segmentation for cutomized data, for models trained on cityscapes',
      url='https://github.com/ddatta-DAC/evalSemanticSeg',
      author='Debanjan Datta',
      author_email='ddatta@cs.vt.com',
      license='MIT',
      packages=['evalSemanticSeg'],
      python_requires='>=3.5',
      install_requires=[
            'numpy',
            'pyyaml',
            'tensorflow>=2.4',
            'pandas',
            'opencv-python>=3.4',
            'scikit-image',
            'scikit-learn'
      ],
      zip_safe=False)
