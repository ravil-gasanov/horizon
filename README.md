# Horizon. A causal discovery algorithm that is not afraid of the dark

This is the implementation of a causal discovery algorithm I developed as part of my master thesis.

Horizon takes in continuous-valued observational data and returns a fully-oriented directed acyclic graph that represents the causal relationships between the variables.

Furthermore, unlike most methods, Horizon does not assume you have measured all relevant variables, and attempts to infer latent confounders as well.