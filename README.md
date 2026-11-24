# Grokking and Optimizers

## Question

A small neural network can **memorize** its training data quickly and only **learn the real rule** much later. This late jump is called *grokking*.

**Does the choice of optimizer, weight decay and training-set size change when that jump happens?**

## Task and setup

- **Task:** modular addition. Given `(a, b)`, predict `(a + b) mod 97`. There are 97 x 97 = 9,409 possible pairs.
- **Split:** a fixed random 30% of pairs for training (2,822) and 70% for testing (6,587), unless stated otherwise. The split is the same in every experiment; only the network's random start (the seed) changes.
- **Model:** a small MLP. Input is two one-hot numbers (194 values), then 256 ReLU hidden units, then 97 output scores (74,849 weights).
- **Training:** full batch, cross-entropy loss, 40,000-step budget per run (150,000 for the long SGD runs). Metrics are logged every 10 steps early on and every 100 steps later. A run stops 1,000 steps after test accuracy first reaches 99%.
- **Optimizers:** AdamW (lr 1e-3), Adam with no weight decay (lr 1e-3), SGD with momentum 0.9 (lr 0.1).
- **Seeds:** every setting was run with 3 seeds.

### Parameters to be measured

| Name | Meaning |
|---|---|
| `T_train` | first step where train accuracy is at least 99% (memorization) |
| `T_test` | first step where test accuracy is at least 99% (generalization) |
| `delay` | `T_test - T_train`, the gap between the two |

A run that never reaches 99% test accuracy has no delay and is reported as "none within N steps".

## Results (3 seeds each)

All runs memorized the training set (train accuracy 1.0) unless noted. 

"Final test acc" is the mean over the three seeds at the end of the run.

**AdamW: weight decay (30% training data, 40,000 steps)**

| Weight decay | Seeds that generalized | Delay in steps, mean (min-max) | Final test acc |
|---|---|---|---|
| 0.1 | 0/3 | none within 40,000 | 0.040 |
| 0.5 | 1/3 | 38,420 (one seed) | 0.988 |
| 1.0 | 3/3 | 21,170 (20,300-21,900) | 0.993 |
| 2.0 | 3/3 | 19,163 (18,660-19,870) | 0.993 |
| 4.0 | 0/3 | none within 40,000 | 0.822 |

At wd = 0.5 the other two seeds were close behind (mean test accuracy over all three: 0.988) when the budget ran out.

**AdamW, wd = 2.0: amount of training data**

| Training fraction | Seeds that generalized | Delay in steps, mean (min-max) | Final test acc |
|---|---|---|---|
| 20% | 0/3 | none within 40,000 | 0.332 |
| 30% | 3/3 | 19,163 (18,660-19,870) | 0.993 |
| 40% | 3/3 | 7,560 (7,290-7,890) | 0.996 |
| 50% | 3/3 | 3,210 (3,080-3,280) | 0.999 |

**Optimizers**

| Experiment | Seeds that generalized | Budget | Final test acc | Notes |
|---|---|---|---|---|
| Adam, wd = 0 | 0/3 | 40,000 | 0.000 | memorized by step ~270 |
| SGD + momentum, wd = 0 | 0/3 | 40,000 | 0.000 | memorized by step ~2,400 |
| SGD + momentum, wd = 1e-3 | 0/3 | 40,000 | 0.032 | memorized by step ~8,400 |
| SGD + momentum, wd = 0.002 | 0/3 | 40,000 | 0.000 | could not fit the training data (train acc 0.33) |
| SGD + momentum, wd = 0.02 | 0/3 | 40,000 | 0.009 | train acc 0.014, essentially chance |
| SGD + momentum, wd = 1e-3 | 0/3 | 150,000 | 0.213 | partial generalization |
| SGD + momentum, wd = 5e-4 | 0/3 | 150,000 | 0.460 | partial generalization |
| SGD + momentum, wd = 3e-4 | 0/3 | 150,000 | 0.480 | partial generalization |

Figures are in `results/`: `overlay_main.png`, `overlay_sgd_long.png`, `sweep_weight_decay.png`, `sweep_train_frac.png`, `delay_all.png`.

## What the results show (in this setup)

1. **Weight decay is necessary for generalization within the budget.** Adam and SGD without weight decay memorized the training set but reached about 0% test accuracy (0 of 6 runs generalized). AdamW with weight decay of 1.0 or 2.0 generalized in every run.
2. **The delay is shortest at moderate weight decay.** Delay was about 21,200 steps at wd = 1.0 and 19,200 at wd = 2.0. Very weak decay (0.1) had not generalized by 40,000 steps, and very strong decay (4.0) memorized later and had reached 82% test accuracy by the end of the budget.
3. **More training data shortens the delay sharply.** The delay fell from about 19,200 steps (30% of the data) to about 7,600 (40%) and about 3,200 (50%), with little spread between seeds. At 20% no run reached 99% within the budget.
4. **SGD with weak weight decay generalizes only partially.** With weight decay of 3e-4 to 1e-3 and 150,000 steps, no SGD run reached 99% test accuracy (best mean 48%). Stronger decay stopped SGD from fitting the training data at all.
