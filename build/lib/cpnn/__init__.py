is_simple_core = False

if is_simple_core:
    from cpnn.core_simple import Variable
    from cpnn.core_simple import Function
    from cpnn.core_simple import using_config
    from cpnn.core_simple import no_grad
    from cpnn.core_simple import as_array
    from cpnn.core_simple import as_variable
    from cpnn.core_simple import setup_variable
else:
    from cpnn.core import Variable
    from cpnn.core import Parameter
    from cpnn.core import Function
    from cpnn.core import using_config
    from cpnn.core import no_grad
    from cpnn.core import as_array
    from cpnn.core import as_variable
    from cpnn.core import setup_variable
    from cpnn.core  import Config
    from cpnn.layers import Layer
    from cpnn.models import Model
    from cpnn.datasets import Dataset
    from cpnn.dataloaders import DataLoader
    from cpnn.dataloaders import SeqDataLoader
    from cpnn.core import test_mode

    import cpnn.datasets
    import cpnn.dataloaders
    import cpnn.optimizers
    import cpnn.functions
    import cpnn.layers
    import cpnn.utils
    import cpnn.cuda
    import cpnn.transforms


setup_variable()
