# Interview guide

## What decision does the project support?

It decides which orders should enter a limited pre-dispatch review queue. The decision is constrained by capacity, so ranking quality at the actionable cutoff matters more than a default 0.5 classification threshold.

## Why use a temporal split?

Order and network behavior changes over time, especially in peak periods. A random split would let observations from later conditions influence development and would overstate stability.

## Why is there a separate calibration period?

The gradient-boosting score is useful for ranking but is not automatically a reliable probability. A later calibration period maps the score to observed risk without fitting that mapping on the final holdout.

## Why did the baseline beat the model on average precision?

The baseline encodes strong operational signals and the holdout is a shifted peak period. The result is evidence against assuming a more complex model wins everywhere. At the fixed 10% decision capacity, however, the model-policy ranking had higher precision and weighted capture. Both facts are reported.

## Is the impact weight a treatment effect?

No. It is a configurable value judgment used for prioritization. The project does not estimate whether any intervention prevents lateness or changes customer behavior.

## What would be required for production?

A point-in-time feature contract, missing-data handling, calibrated monitoring, queue feedback, policy approval, intervention-cost constraints, fallback rules, access controls, and experimental evidence for each proposed treatment.

