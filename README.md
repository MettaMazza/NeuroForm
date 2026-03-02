# NeuroForm

**NeuroForm** is a lightweight, fully autonomous Neo4j memory graph system designed specifically for local LLMs (via Ollama). 

Instead of relying *solely* on hardcoded mathematical heuristics, NeuroForm utilizes a **Hybrid Heuristic + LLM** engine to grant the local LLM direct, higher-order semantic control over its own memory graph. 

## Features

- **Hybrid Neuroplasticity (Math + LLM)**: 
  - **Phase 1: Homeostasis**: Graph connections naturally decay over time based on `last_fired` timestamps. Connections that hit 0 are pruned mathematically (biological noise reduction).
  - **Phase 2: Semantic Reasoning**: The memory loop (`neuroplasticity.py`) fetches sub-graphs of surviving memories and feeds them to the LLM. The LLM issues structured JSON commands to semantically `PRUNE` contradictions, `STRENGTHEN` critical identity traits, or intentionally `DECAY` fading relevance.
- **Neo4j Graph Backend**: Scalable, structured entity-relationship mapping.
- **Local First**: Built primarily to interface with Ollama (defaults to `llama3`) for complete privacy and local execution.
- **100% Test Coverage**: The core logic is fully covered by parameterized `pytest` unit suites utilizing mock database drivers for rapid validation.

## Architecture

1. **`KnowledgeGraph` (`graph.py`)**: The low-level Neo4j driver wrapper. Handles connection pooling, ErnOS-layer 6-tier mesh topology routing, and Cypher query execution for CRUD operations. It tracks `last_fired` timestamps to simulate biological action potentials.
2. **`OllamaClient` (`ollama_client.py`)**: The interaction layer. Fetches relevant memory context from the graph, injects it into the LLM prompt, and then extracts new memories from the LLM's response to store back into Neo4j.
3. **`AutonomousNeuroplasticity` (`neuroplasticity.py`)**: The background daemon loop. Combines baseline temporal mathematical decay with high-level LLM intellectual review. The LLM decides what memories are obsolete, contradictory, or vital.
4. **`AgencyDaemon` & `ToolManager` (`daemons/`, `tools/`)**: A continuous, interruptible background process and tool execution engine. Allows Nero to execute multi-turn autonomous loops using native OS tools (shell execution, AppleScript, filesystem ops) constrained by a two-tier permission system (OWNER vs. SAFE).

## Installation

1. Confirm you have Python 3.9+ installed.
2. Clone the repository.
3. Install the package via pip:
   ```bash
   pip install -e .
   ```
4. Copy `.env.example` to `.env` and fill in your Neo4j credentials, Ollama model, and Discord bots variables:
   - `DISCORD_TOKEN` and `DISCORD_CHANNEL_ID` (your main chat channel)
   - `DISCORD_AUTONOMY_CHANNEL_ID` (optional, routes background thoughts away from the main chat)
   - `DISCORD_OWNER_ID` (comma-separated Discord user IDs allowed to run OS-level tools)
5. Ensure `ollama` is running locally with your desired model (e.g., `ollama pull llama3`).

## Documentation & Science

For a deep dive into how NeuroForm works under the hood and the cognitive science it is based on, refer to the `docs/` folder:

1. [The Cognitive Science of NeuroForm](docs/neuroplasticity_science.md) - Explains biological neuroplasticity, Hebbian learning, and the computational analogy (strengths and limits).
2. [Architectural Overview](docs/architecture.md) - Details the 3 core pillars (Interaction, Storage, and Autonomous Optimization).
3. [API Reference](docs/api_reference.md) - Method-level documentation for developers.

## Usage

You can run the interactive standalone boot script to see it in action:

```bash
python -m neuroform.main
```

1. **Chat**: You can chat directly with the agent. Behind the scenes, it will pull context from Neo4j and intelligently extract new facts to store.
2. **Optimization Loop**: On boot (or on a cron schedule in production), the system will run `evaluate_and_optimize()`. Watch as the LLM prunes outdated connections or strengthens core identity facts autonomously.

## Development & Testing

We enforce strict 100% test coverage on core logical components.

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=neuroform
```

## License

MIT License. See `LICENSE` for details.
