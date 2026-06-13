from setuptools import setup

package_name = "yolo_detector"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="team2",
    maintainer_email="daniel-hinojosa09@outlook.com",
    description="YOLO detection node publishing an annotated compressed image.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_node = yolo_detector.yolo_node:main",
        ],
    },
)
