# Third-party components

- [SpecForge](https://github.com/sgl-project/SpecForge), MIT license. The `specforge`
  package and `alr/train_backend.py` are adapted from its training implementation.
  See LICENSE.SpecForge.
- [DFlash](https://github.com/z-lab/dflash), MIT license. `dflash/model.py` provides
  the public draft architecture and text decoder. See LICENSE.DFlash.

The ALR and IRA objectives, the target rollout path and the vision verification
utilities extend these components. Upstream copyright notices are retained. Model and
dataset licenses remain with their respective distributors. The gpt-oss chat templates and
the Harmony parser of SpecForge are removed, since no run in the paper uses them.
