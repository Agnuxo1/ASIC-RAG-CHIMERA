# Lucky Miner LV06: Programmer's Guide to Neuromorphic Computing

## 1. Introduction
The Lucky Miner LV06, powered by Bitmain's **BM1387** or **BM1366** ASICs, is a powerful tool for Neuromorphic Reservoir Computing. While designed for SHA-256 mining, its hardware architecture can be repurposed as a high-entropy physical reservoir for chaotic time-series prediction and complex non-linear mapping.

## 2. System Architecture
The device consists of two main components:
*   **The Controller (ESP32):** Manages the web interface, WiFi/Ethernet, and the Stratum connection. It parses incoming jobs and schedules them for the ASIC.
*   **The ASIC (BM1387/1366):** The computational core. It performs quadrillions of hash operations. Any variation in its timing or nonce selection (Jitter) is the basis of our neuromorphic state.

## 3. The Reservoir Computing Workflow
To use the LV06 as a reservoir, we follow the **HRC (Holographic Reservoir Computing)** paradigm:

1.  **Input Injection (Rate Encoding):** We encode training data into the **Mining Difficulty**.
2.  **Stratum Notification:** We send `mining.set_difficulty` instructions followed by a `mining.notify` heartbeat.
3.  **State Harvesting:** The miner returns "Shares". The **arrival density** (Poisson rate) of these shares is directly modulated by the input.
4.  **Feature Extraction:** We use the count of shares and the inter-arrival times (IAT) as the high-dimensional state vector.
5.  **Training:** We use Ridge Regression to map these physical states back to our target values.

## 4. Key Performance Drivers
### Dynamic Difficulty (Rate Control)
*   **Modulation Range:** Recommended $D_{base}$ is 25.0 for 1Hz sampling.
*   **Poisson Statistics:** The variance in share arrivals ($\sigma^2$) provides the stochastic entropy needed for complex prediction tasks.

## 5. Best Practices
*   **Handshake Timing:** Always wait for the `mining.authorize` confirmation before injecting data.
*   **Cumulative Harvesting:** Instead of waiting for a specific time window, always process the entire buffer of received shares to avoid network jitter artifacts.
*   **Thermal Safety:** Monitor temperatures via the `/api/status` endpoint to prevent hardware degradation.

---
*Created by the ASIC-RAG-CHIMERA Research Team.*
