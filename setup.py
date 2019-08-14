import setuptools
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext as DistUtilsBuildExt


class BuildExtension(setuptools.Command):
    description     = DistUtilsBuildExt.description
    user_options    = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options    = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy
        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


extensions = [
    Extension(
        'keras_m2det.utils.compute_overlap',
        ['keras_m2det/utils/compute_overlap.pyx']
    ),
]


setuptools.setup(
    name             = 'keras-m2det',
    version          = '0.5.1',
    description      = 'Keras implementation of m2det object detection.',
    url              = 'https://github.com/LeeDongYeun/keras-m2det',
    author           = 'LeeDongYeun',
    author_email     = 'ledoye@kaist.ac.kr',
    maintainer       = 'LeeDongYeun',
    maintainer_email = 'ledoye@kaist.ac.kr',
    cmdclass         = {'build_ext': BuildExtension},
    packages         = setuptools.find_packages(),
    install_requires = ['keras', 'keras-resnet==0.1.0', 'six', 'scipy', 'cython', 'Pillow', 'opencv-python', 'progressbar2'],
    entry_points     = {
        'console_scripts': [
            'm2det-train=keras_m2det.bin.train:main',
            'm2det-evaluate=keras_m2det.bin.evaluate:main',
            'm2det-debug=keras_m2det.bin.debug:main',
            'm2det-convert-model=keras_m2det.bin.convert_model:main',
        ],
    },
    ext_modules    = extensions,
    setup_requires = ["cython>=0.28", "numpy>=1.14.0"]
)
