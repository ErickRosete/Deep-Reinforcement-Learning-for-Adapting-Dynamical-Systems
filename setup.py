from setuptools import setup, find_packages

setup(
      name='drlfads',
      version='0.0.1',
      description='Utilizes Deep RL to adapt dynamical systems',
      author='Erick Rosete Beas',
      author_email='erickrosetebeas@hotmail.com',
      url='https://aisgit.informatik.uni-freiburg.de/erosete/Deep-Reinforcement-Learning-for-Adapting-Dynamical-Systems',
      packages=find_packages(),
      install_requires=['pybulletX',
                'hyperopt',
                'gym(==0.18.0)',
                'hpbandster(==0.7.4)',
                'hydra.core(==1.0.6)',
                'tacto(==0.0.2)',
                'tensorboard(==2.4.1)',
                'torch(==1.7.1)',
                'matlab(>=0.1)']
     )
