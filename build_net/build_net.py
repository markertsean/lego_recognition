import os
import imp

# Get the path to the lego directory
_package_path = os.path.abspath(os.path.dirname(__file__))
_package_path = _package_path[:-9]


# Load the image processing library
_ip_subpkg = 'image_processing'
_ip_module = 'image_processing.py'
ip         = imp.load_source(_ip_subpkg, _package_path+_ip_subpkg+'/'+_ip_module)
