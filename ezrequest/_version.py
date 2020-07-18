import os

__version__: str = ""
__version__year__: int = -1
__version__doy__: int = -1
__version__minute__:int = -1

if os.path.isfile('.version.txt'):
    with open('.version.txt', 'r') as fh:
        for line in fh.readlines():
            if not line.startswith('#'):
                __version__ = fh.readline()
                version_parts = __version__.split('.')
                __version__year__ = int(version_parts[0])
                __version__doy__ = int(version_parts[1])
                __version__minute__ = int(version_parts[2])
else:
    from datetime import datetime
    now = datetime.now()
    __version__year__ = now.year
    __version__doy__ = (now - datetime(now.year, 1, 1)).days
    __version__minute__ = now.hour*60 + now.minute
    __version__ = '.'.join([str(__version__year__), str(__version__doy__), str(__version__minute__)])
