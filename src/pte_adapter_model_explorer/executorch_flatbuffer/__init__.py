from .program_generated import *
from .scalar_type_generated import *
from .xnnpack_generated import *

# flatc emits XnodeUnionCreator/XvalueUnionCreator (lowercase 'n' and 'v') but 
# generated classes call XNodeUnionCreator/XValueUnionCreator (uppercase 'N' and 'V') 
# at runtime; add aliases once here so we don't touch generated files.
from . import xnnpack_generated as _xnnpack 

def _alias_creator(alias_name, *candidate_names):
    if hasattr(_xnnpack, alias_name):
        target = getattr(_xnnpack, alias_name)
    else:
        target = None
        for candidate_name in candidate_names:
            if hasattr(_xnnpack, candidate_name):
                target = getattr(_xnnpack, candidate_name)
                setattr(_xnnpack, alias_name, target)
                break

    if target is not None and alias_name not in globals():
        globals()[alias_name] = target

_alias_creator("XNodeUnionCreator", "XnodeUnionCreator")
_alias_creator("XValueUnionCreator", "XvalueUnionCreator")
_alias_creator("XNNQuantParamsCreator", "XnnquantParamsCreator", "XnnQuantParamsCreator")

del _xnnpack
