from setuptools import setup

setup(
    name='IS623_ImageTools',
    version='1.0.0',
    packages=['IS623_ImageTools'],
    description="A simple library for image processing and visualization for the IS623 - Computación Gráfica course at Universidad Tecnológica de Pereira (UTP), developed by Kevin Esguerra Cardona.",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Kevin Esguerra Cardona',
    author_email='kevin.esguerra@utp.edu.co',
    url='https://github.com/porgetit/IS623-ImageTools.git',
    install_requires=[
        "numpy",
        "matplotlib",
        "Pillow"
    ]
    
)