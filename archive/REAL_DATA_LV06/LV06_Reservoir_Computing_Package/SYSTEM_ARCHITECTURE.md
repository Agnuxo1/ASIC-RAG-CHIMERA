# HRC System Architecture Overview

The Holographic Reservoir Computing (HRC) system on the Lucky Miner LV06 operates by exploiting the stochastic nature of the Bitcoin mining hashing pipeline.

## 1. Input Layer: Rate-Encoding (RE-RC)
Input signals $u(t)$ are injected into the system by modulating the **Difficulty** parameter of the Stratum protocol. 
$$D(t) = \frac{D_{base}}{u(t) + \epsilon}$$
This creates a physical coupling where the expected rate of share arrivals is proportional to the input.

## 2. Reservoir Layer: The BM1366 ASIC
The ASIC chip functions as a non-linear high-dimensional dynamical system. The "State Transfer" occurs within the hashing pipeline, where the inter-arrival times (IAT) of discovered nonces represent the reservoir's instantaneous response.

## 3. Readout Layer: Ridge Regression
The output features are extracted from the share distribution within a fixed temporal window ($T = 2000ms$):
- Share Count
- Mean Inter-Arrival Time (IAT)
- Std Dev of IAT
- Coefficient of Variation (CV)

A linear readout layer (Ridge Regression) is trained to map these hardware features back to the target computational task (e.g., NARMA-10).

## 4. Stability Protocol: "Cool Reservoir"
To maintain hardware honesty and prevent thermal decoupling, the system uses:
- Underclocking (450MHz) to reduce jitter.
- Over-voltage (1100mV) relative to clock to ensure signal integrity in the pipeline.
- Large sampling windows to capture the Poisson dynamics accurately.
