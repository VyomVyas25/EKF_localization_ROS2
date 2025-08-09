import os
from setuptools import find_packages, setup

package_name = 'py_pubsub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vyom',
    maintainer_email='vyom@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'localization = py_pubsub.EKF_localization:main',
            'm2C = py_pubsub.m2C:main',
            'gd = py_pubsub.Ground_det:main',
            'ekf_slam = py_pubsub.EKF_slam:main',
            'slam_d = py_pubsub.ekf_slam_d:main',
            'binary_im = py_pubsub.gnd_obj_det:main',

        ],
    },
)
