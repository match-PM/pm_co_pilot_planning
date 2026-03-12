from setuptools import find_packages, setup

package_name = 'pm_co_pilot_planning'
submodules = 'pm_co_pilot_planning/submodules'
files = 'files'
langchain = 'pm_co_pilot_planning/submodules/langchain'
tools = 'pm_co_pilot_planning/submodules/langchain/tools'

setup(
    name=package_name,
    version='0.0.0',
    # packages=find_packages(exclude=['test']),
    packages=(package_name, submodules, tools, files, langchain),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['config/blacklist.yaml']),
        ('share/' + package_name, ['config/whitelist.yaml']),
        ('share/' + package_name, ['config/Prompts.yaml']),
        ('share/' + package_name, ['config/assembly_config.yaml']),
        ('share/' + package_name, ['config/service_registry.yaml']),
        ('share/' + package_name + '/launch', ['launch/pm_co_pilot_planning.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='match-mover',
    maintainer_email='wiemann@match.uni-hannover.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pm_co_pilot_planning = pm_co_pilot_planning.pm_co_pilot_planning:main'
        ],
    },
)
