from setuptools import setup, find_packages


setup(
    name='trafficsignrecognition',
    version='0.1',
    description="Bengali Traffic Sign Detection and Recognition",
    author='Omar Faruk Sazib',
    author_email='ofsazib@gmail.com',
    packages=find_packages(),
    install_requires=['menpofit>=0.4,<0.5',
                      'menpowidgets>=0.2,<0.3']
)
