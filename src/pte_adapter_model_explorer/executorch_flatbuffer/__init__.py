from .program_generated import *
from .scalar_type_generated import *
from .xnnpack_generated import *

# flatc emits XnodeUnionCreator/XvalueUnionCreator (lowercase 'n' and 'v') but 
# generated classes call XNodeUnionCreator/XValueUnionCreator (uppercase 'N' and 'V') 
# at runtime; add aliases once here so we don't touch generated files.
from . import xnnpack_generated as _xnnpack 

if not hasattr(_xnnpack, "XNodeUnionCreator") and hasattr(_xnnpack, "XnodeUnionCreator"):
    setattr(_xnnpack, "XNodeUnionCreator", getattr(_xnnpack, "XnodeUnionCreator"))

if not hasattr(_xnnpack, "XValueUnionCreator") and hasattr(_xnnpack, "XvalueUnionCreator"):
    setattr(_xnnpack, "XValueUnionCreator", getattr(_xnnpack, "XvalueUnionCreator"))

del _xnnpack
