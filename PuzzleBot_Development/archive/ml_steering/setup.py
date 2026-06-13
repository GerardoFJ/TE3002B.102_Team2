import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ml_steering'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Ship the trained model with the package so the runtime node finds
        # it via get_package_share_directory(). Offline `train_xgboost.py`
        # writes to models/ in the source tree; colcon copies it to share/.
        (os.path.join('share', package_name, 'models'),
            glob('models/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='puzzlebot',
    maintainer_email='puzzlebot@todo.todo',
    description='XGBoost steering predictor for the Puzzlebot vision pipeline.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ml_steering_node = ml_steering.ml_steering_node:main',
        ],
    },
)
