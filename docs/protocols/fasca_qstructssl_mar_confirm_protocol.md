# FASCA-QStructSSL-MAR confirmation protocol

The method and all hyperparameters were frozen after the seed-51 five-fold
screen. Confirmation adds student seeds 52 and 53 on the same five fixed
subject-disjoint folds. No parameter is selected from their test results.

The primary endpoint is the 55-subject paired mean over the five standard
joint-fault scenarios. Secondary endpoints are the all-corruption mean,
15-combination missing-modality mean, worst modality combination, and clean
full-modality performance.

The candidate is considered to have a meaningful advantage when the primary
endpoint remains positive with a 95% paired bootstrap interval excluding zero
after pooling all three seeds, while clean performance and the missing-modality
mean do not show a statistically supported deterioration.
