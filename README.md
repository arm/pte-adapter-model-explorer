# PTE Adapter for Model Explorer

PTE Adapter for [google-ai-edge/model-explorer](https://github.com/google-ai-edge/model-explorer) that enables visualization of PTE files for
[Arm® Ethos™-U NPU Backend](https://docs.pytorch.org/executorch/stable/backends-arm-ethos-u.html) and [Arm® VGF Backend](https://docs.pytorch.org/executorch/stable/backends-arm-vgf.html) targets.


## Requirements

- Python >=3.10, <3.13

## Supported Platforms

- Linux x86_64
- Linux AArch64
- MacOS AArch64
- Windows x86_64

## Installation

### pip + PyPI
    pip install pte-adapter-model-explorer

### GitHub

    PYTHON_VERSION_TAG=310 &&
    gh release download \
    --repo arm/pte-adapter-model-explorer \
    --pattern "*py${PYTHON_VERSION_TAG}*.whl" &&
    pip install *py${PYTHON_VERSION_TAG}*.whl

Or through the [GitHub Releases](https://github.com/arm/pte-adapter-model-explorer/releases) UI.

## Usage

Install Model Explorer:

    pip install torch ai-edge-model-explorer

Launch Model Explorer with the PTE adapter enabled:

    model-explorer --extensions=pte_adapter_model_explorer

See the [Model Explorer wiki](https://github.com/google-ai-edge/model-explorer/wiki) for more information.

## Trademark notice

This project uses some of the Arm® product, service or technology trademarks, as listed in the [Trademark List](https://www.arm.com/company/policies/trademarks/arm-trademark-list), in accordance with the Arm [Trademark Use Guidelines](https://www.arm.com/company/policies/trademarks/guidelines-trademarks).

Subsequent uses of these trademarks throughout this repository do not need to be prefixed with the Arm word trademark.

## Contributions

We are not accepting direct contributions at this time.
If you have any feedback or feature requests, please use the repository issues section.
