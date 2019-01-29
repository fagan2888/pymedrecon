
# pymedrecon

A set of tools for medical image reconstruction in Python

## Documentation

Full docstring comments are provided for many functions. It is recommended to
view documentation in ipython. To view documentation, first, launch ipython.
Then, type

```python
help(function.py)
```

Any documentation for that function should be displayed in the terminal.

## List of projects

- mri: tools for magnetic resonance imaging reconstruction
  - systems.mri: a Python object for forward and adjoints with senstivity coils
  - systems.dft: a Python object that is essentially a DFT wrapper
  - systems.kbnufft: a Python object for non-uniform FFTs in numpy
  - mrisensesim: sensitivity coil simulation
  - calc_smap_espirit.py: estimation of sensitivity coils via ESPIRiT

## Test scripts

Test scripts are implemented as __main__ functions. Not all scripts have test
functions.