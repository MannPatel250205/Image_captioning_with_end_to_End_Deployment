from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path : str) -> List[str]:
    """
    This function will return the list of requirements
    """
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
    
    # Removing any empty lines and stripping whitespace
    requirements = [req.strip() for req in requirements if req.strip()]
    if "-e ." in requirements:
        requirements.remove("-e .")
    return requirements


setup(
    name="Image_captioning_end_to_end",
    version="0.0.1",
    author="Mann",
    author_email="mannpatel7744@gmail.com",
    description="Image captioning end to end using Tensorflow and Keras",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)