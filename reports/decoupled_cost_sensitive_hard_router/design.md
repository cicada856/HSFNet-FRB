# DLSR Technical Design and Outcome

## Difference From the Paper System

The paper routes samples to the local specialists when the five-class base
Top-1 prediction is B, C1, or C2. DLSR separates classification from routing:

```text
Frozen ResNet50 global-average-pooled feature + base probabilities
    -> binary global(A/C) versus local(B/C1/C2) router
    -> discrete hard route
    -> unchanged specialists and beta=0.20 fusion
```

The five-class base and all specialist checkpoints remain frozen. The router is
an inference-time component rather than a training-only auxiliary head.

## Cost-Sensitive Training

The router uses weighted binary cross-entropy or focal binary cross-entropy.
False global-to-local routes and false local-to-global routes have separately
selectable costs. CHIME development examples receive an additional domain
weight relative to simulated examples.

Six router configurations were compared. Each was evaluated with five-fold OOF
predictions on the immutable 267-sample CHIME development half. Three hard
policies were searched:

- router-only routing;
- base/router dual confirmation;
- dual confirmation with high-confidence rescue of uncertain base A/C outputs.

The current C1 route is explicitly protected. Paper-test samples were not
indexed until the router configuration and thresholds had been locked.

## Locked Configuration

- 64-unit hidden layer and 32-unit bottleneck;
- focal gamma 2.0;
- global false-positive cost 2.0;
- local false-negative cost 1.0;
- CHIME development weight 16.0;
- dual-confirm policy;
- local confirmation threshold 0.40;
- base-predicted C1 always retains the current specialist route.

The router contains 133,825 parameters and adds approximately 0.225 ms per
sample after the already-computed base feature is available.

## Development OOF Result

Accuracy improved from 58.05% to 58.43% and macro-F1 from 46.28% to 47.04%.
The policy changed 14 routes and two final predictions, with one correction,
zero harmful changes, and one wrong-to-different-wrong change. C1 recall remained
20%. This passed the predeclared development guard.

## Locked Paper-Test Result

Across seeds 42, 43, and 44, mean accuracy was 63.18% versus 63.43% for the
paper system. Mean macro-F1 was 53.07% versus 53.35%. C1 recall remained 30%,
and A, C, and C2 recall were unchanged.

Seed 42 produced the same route and prediction set as the paper system. Seeds
43 and 44 rejected the specialist for sample 525. Its base prediction was C2,
while the current specialist fusion correctly changed it to B. DLSR therefore
changed this correct B result back to C2, causing one harmful change in each of
those seeds.

The result satisfies the numerical accuracy and macro-F1 tolerance but fails
the stability and favorable-change requirements. DLSR is a valid active hard
routing ablation, but it should not replace the current Top-1 route.
