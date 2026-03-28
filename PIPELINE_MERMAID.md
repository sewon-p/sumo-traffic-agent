# Pipeline Diagram Reference

This document contains standalone Mermaid versions of the main pipeline diagrams. The primary version now lives in [README.md](README.md).

Key points:

- the fine-tuned model converts natural language into structured parameters
- the base LLM handles geometry and XML reasoning
- SUMO acts as the synthetic scenario execution engine
- user interactions are stored as either `correction` or `tuning`
- only `correction` records are exported as retraining signals by default

## 1. End-to-End Generation Pipeline

```mermaid
flowchart TD
    A[User Natural-Language Request] --> B[FT Parameter Extraction<br/>src/llm_parser.py]
    B --> B1[Structured Scenario Parameters<br/>speed_kmh / volume_vph / lanes / speed_limit / sigma / tau / avg_block_m]

    B1 --> C{Location / Map Available?}
    C -- Yes --> D[OSM-Based Network Retrieval<br/>tools/osm_network.py]
    C -- No or Failed --> E[Base LLM Geometry/XML Generation<br/>src/base_llm.py]

    D --> F[SUMO Network Build<br/>netconvert]
    E --> F

    B1 --> G[Demand / Route / Config Generation<br/>tools/sumo_generator.py]
    F --> H[Runnable SUMO Artifacts<br/>.net.xml / .rou.xml / .add.xml / .sumocfg]
    G --> H

    H --> I[SUMO Scenario Execution]
    I --> J[Validation / Stats Extraction<br/>src/validator.py]
    J --> K[Simulation Output + Evaluation Summary]
```

## 2. Human-in-the-Loop Refinement Pipeline

```mermaid
flowchart TD
    A[Generated Scenario Result] --> B[User Interaction]
    B --> C{Interaction Type}

    C -- Correction --> D[Trainable Feedback]
    C -- Tuning --> E[Non-Trainable Variant Request]

    D --> F{Modification Target}
    E --> F

    F -- Parameter --> G[Base LLM Parameter Update]
    F -- Geometry --> H[Base LLM Geometry/XML Update]
    F -- Mixed --> I[Parameter Update + Geometry Update]

    G --> J[Rebuild / Re-run SUMO]
    H --> J
    I --> J

    J --> K[Updated Scenario]
    K --> L[Store Session + Modification Log<br/>src/session_db.py]
```

## 3. Retraining Data Pipeline

```mermaid
flowchart TD
    A[Simulation Session] --> B[Store Initial Prompt]
    A --> C[Store FT Output]
    A --> D[Store Final Params]
    A --> E[Store Prompt Metadata]
    A --> F[Store XML Snapshots]
    A --> G[Store Modification Events]

    G --> H{edit_intent}
    H -- correction --> I[trainable = 1]
    H -- tuning --> J[trainable = 0]

    I --> K[Correction-Only Export<br/>src/session_db.py]
    J --> L[Keep for Analysis Only]

    K --> M[OpenAI JSONL / Future FT Dataset]
    I --> N[Field Error Counts]
    I --> O[Average Parameter Delta]
    I --> P[Geometry Correction Categories]
```

## 4. Model / Tool Role Separation

```mermaid
flowchart LR
    A[User Request] --> B[Fine-Tuned Model]
    B --> C[Structured Parameters]

    A --> D[Base LLM]
    D --> E[Geometry Reasoning]
    D --> F[XML Generation / Modification]
    D --> G[Modification Type Classification]

    C --> H[Tool Layer]
    E --> H
    F --> H
    G --> H

    H --> I[OSM / SUMO / DB / Validation]
```

## 5. Admin / Evaluation Pipeline

```mermaid
flowchart TD
    A[Simulation Logs] --> B[Admin Backend APIs<br/>server.py]
    C[Modification Logs] --> B
    D[Prompt Metadata] --> B

    B --> E[Admin Dashboard<br/>web/admin.html]
    B --> F[Correction Export Download]
    B --> G[Evaluation Report Download]
    B --> H[LLM Evaluation Report Download]

    E --> I[Correction Rate]
    E --> J[Field Error Counts]
    E --> K[Average Parameter Delta]
    E --> L[Geometry Correction Categories]
```

## 6. Full Closed Loop

```mermaid
flowchart TD
    A[Prompt / Scene Description] --> B[FT Extraction]
    B --> C[Scenario Build]
    C --> D[SUMO Execution]
    D --> E[Validation]
    E --> F[Human Correction / Tuning]
    F --> G[Structured Logging]
    G --> H[Error Analysis / Admin Reports]
    H --> I[Correction-Only Export]
    I --> J[Future Fine-Tuning Dataset]
    J --> B
```
