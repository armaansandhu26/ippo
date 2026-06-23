# Collapse Summary (dense history)

| Model | Biased train | Collapse@0.75 | Collapse@0.90 | Collapse@0.95 | Final validate A-rate | Final train-minus-unbiased-validate A-rate gap |
|-------|--------------|---------------|---------------|---------------|-----------------------|-----------------------------------------------|
| gemma3-1b | no | 10±0 | 20±10 | 30±0 | 0.844±0.068 | 0.021±0.039 |
| gemma3-1b | yes | 10±0 | 17±12 | 53±12 | 0.922±0.041 | 0.036±0.055 |
| gemma3-4b | no | — | — | — | 0.328±0.016 | -0.089±0.039 |
| gemma3-4b | yes | — | — | — | 0.349±0.024 | 0.568±0.009 |
