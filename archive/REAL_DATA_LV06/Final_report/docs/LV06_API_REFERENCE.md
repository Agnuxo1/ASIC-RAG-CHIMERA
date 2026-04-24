# Lucky Miner LV06 API Reference

This document provides the technical specifications for the LV06 (AxeOS) Stratum and HTTP APIs.

## 1. Stratum Protocol (TCP/3333)
The miner acts as a Stratum client. The following methods are supported:

### `mining.subscribe`
Initializes the connection.
*   **Request:** `{"id": 1, "method": "mining.subscribe", "params": ["AgentName"]}`
*   **Response:** `{"id": 1, "result": [ [ ["mining.set_difficulty", "sub_id_1"], ["mining.notify", "sub_id_2"] ], "extranonce1", extranonce2_size], "error": null}`

### `mining.authorize`
Authorizes the worker.
*   **Request:** `{"id": 2, "method": "mining.authorize", "params": ["worker_name", "password"]}`
*   **Response:** `{"id": 2, "result": true, "error": null}`

### `mining.notify` (Server to Client)
Sends a heartbeat to trigger the new difficulty setting.
*   **Params:** `[job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean_jobs]`
*   **Trigger:** Set `clean_jobs` to `true` to ensure the chip immediately starts on the new difficulty-modulated job.

## 2. Rate-Encoded Injection
For high-fidelity data injection, use the following logic:
1.  **Calculate Difficulty:** $D = D_{base} / (u[t] + \epsilon)$.
2.  **Send Request:** `mining.set_difficulty(D)`.
3.  **Halt & Reset:** `mining.notify(clean_jobs=True)`.
