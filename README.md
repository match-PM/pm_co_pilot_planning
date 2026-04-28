# PM Co-Pilot Planning

An AI-powered co-pilot for micro-assembly sequence planning and execution in ROS2. It provides a conversational interface — voice or text — to build, modify, and run action sequences on robotic systems using LLM-based agents with persistent knowledge management.

---

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Configuration](#configuration)
- [Launch](#launch)
- [Architecture](#architecture)
- [Usage](#usage)

---

## Dependencies

### System Requirements

- ROS2 (Humble or later)
- Python 3.10+
- PyQt6
- `portaudio` (for microphone input)

### ROS2 Packages

- `ros_sequential_action_programmer` — sequence execution framework (must be built in the same workspace)
- `assembly_manager_interfaces` — message types for assembly state

### Python Packages

Install via pip (inside your ROS2 environment):

```bash
pip install langchain langgraph langchain-openai langchain-anthropic langchain-community \
            openai anthropic pydub PyQt6 speech_recognition
```

| Package | Purpose |
|---|---|
| `langchain` / `langgraph` | LLM agent framework and ReAct graph execution |
| `langchain-openai` | OpenAI model provider |
| `langchain-anthropic` | Anthropic (Claude) model provider |
| `langchain-community` | Community integrations |
| `openai` | Whisper (speech-to-text) and TTS (text-to-speech) |
| `anthropic` | Direct Anthropic API access |
| `PyQt6` | GUI framework |
| `speech_recognition` | Microphone audio capture |
| `pydub` | Audio format conversion for Whisper |

### API Keys

Set the relevant keys in your environment before launching:

```bash
export OPENAI_API_KEY="sk-..."          # required for speech I/O (Whisper + TTS)
export ANTHROPIC_API_KEY="sk-ant-..."   # required if using Claude models (default)
# export GOOGLE_API_KEY="..."           # optional, for Gemini models
```

---

## Installation

```bash
# 1. Clone into your ROS2 workspace
cd ~/ros2_ws/src
git clone <repo-url> pm_co_pilot_planning

# 2. Install Python dependencies
pip install langchain langgraph langchain-openai langchain-anthropic langchain-community \
            openai anthropic pydub PyQt6 speech_recognition

# 3. Build the workspace
cd ~/ros2_ws
colcon build --packages-select pm_co_pilot_planning

# 4. Source the workspace
source install/setup.bash
```

---

## Configuration

All configuration lives in [`config/`](config/).

### `config/Prompts.yaml`

The primary configuration file. Controls which LLM is used for each agent phase and contains the system prompts.

```yaml
planner:
  model: 'claude-haiku-4-5'     # LLM model name
  model_provider: 'anthropic'   # anthropic | openai | google | xai
  temperature: 0.0
  system_prompt: >
    ...

executor:
  model: 'claude-haiku-4-5'
  ...

learner:
  model: 'claude-haiku-4-5'
  ...

consolidator:
  model: 'claude-haiku-4-5'
  ...
```

**Available models** (see `available_models` section in the file):

| Provider | Models |
|---|---|
| Anthropic | `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-6` |
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-4` |
| Google | `gemini-2.0-flash-001`, `gemini-1.5-flash` |

### `config/assembly_config.yaml`

Paths to the component and assembly databases used by the assembly knowledge tools.

### `config/whitelist.yaml` / `config/blacklist.yaml`

Filter which ROS2 services are visible to the agent.

---

## Launch

```bash
ros2 launch pm_co_pilot_planning pm_co_pilot_planning.launch.py
```

Or run directly:

```bash
ros2 run pm_co_pilot_planning pm_co_pilot_planning
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    PyQt6 GUI                        │
│  Chat Display │ Text Input │ Listen │ Send          │
│  Record Knowledge ☐  │  Consolidate Knowledge ☐    │
└───────────────────────┬─────────────────────────────┘
                        │ MessageWorker (QThread)
                        ▼
┌─────────────────────────────────────────────────────┐
│                    Agent (Orchestrator)              │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │              PhaseController                │   │
│  │  planning → executing → escalated           │   │
│  │                       → learning            │   │
│  └──────────────┬──────────────────────────────┘   │
│                 │                                   │
│  ┌──────────────▼──────────────────────────────┐   │
│  │           ExecutionMonitor                  │   │
│  │  LangGraph ReAct agent, one invocation      │   │
│  │  Streams tool calls + responses             │   │
│  └──────────────┬──────────────────────────────┘   │
│                 │                                   │
│  ┌──────────────▼──────────────────────────────┐   │
│  │              Tool Sets                      │   │
│  │  Planner:   KB + Assembly DB + Sequence     │   │
│  │  Executor:  Execution + parameter fixing    │   │
│  │  Learner:   query / record / confirm KB     │   │
│  └──────────────┬──────────────────────────────┘   │
└─────────────────┼───────────────────────────────────┘
                  │
     ┌────────────┴──────────────┐
     ▼                           ▼
RSAP Framework              Knowledge Base
(ROS2 service calls)        service_knowledge.yaml
```

### Agent Phases

| Phase | Model | Responsibility |
|---|---|---|
| **planning** | Planner LLM | Understand request, design sequence, handle user Q&A |
| **executing** | Executor LLM | Run one action at a time, fix parameters on failure |
| **escalated** | Planner LLM | Diagnose complex errors during execution |
| **learning** | Learner LLM | Record service usage notes and parameter descriptions |
| **consolidation** | Consolidator LLM | Merge and deduplicate the knowledge base |

### Tool Groups

**Sequence tools** (`RsapTools.py`) — interact with the RSAP sequence:
- `get_available_services` — list callable ROS2 services
- `build_sequence_from_plan` — batch-add a list of actions
- `add_service_to_sequence` / `delete_action` / `move_action` — atomic edits
- `set_action_parameters` — update parameters of one action
- `execute_single_action` — run one action and capture the result
- `save_sequence` / `load_sequence` / `clear_sequence` — persistence

**Knowledge tools** (`KnowledgeTools.py`) — manage service knowledge base:
- `query_assembly_knowledge` — fetch recorded notes and parameters for a service
- `record_knowledge` — save a new usage note or parameter description
- `confirm_knowledge` / `contradict_knowledge` — rate existing notes

**Assembly tools** (`AssemblyKnowledgeTools.py`) — query assembly database and scene state:
- `list_available_components` / `get_component_description` — component DB
- `list_available_assemblies` / `get_assembly_description` — assembly DB
- `list_objects_in_scene` / `get_object_frames` / `get_frames_in_scene` — live scene state

### Knowledge Management

The agent maintains a `service_knowledge.yaml` file that accumulates usage notes and parameter descriptions across sessions. After each execution, the **learning phase** records what happened. Every 3 learning sessions (when enabled), the **consolidation phase** merges duplicate entries and resolves contradictions.

### Session Logs

At the end of each session, the full interaction history is saved to `~/Desktop/copilot_log_<timestamp>.json`. Each log contains:
- All messages and tool calls with timestamps and token counts
- The final action sequence
- User feedback (success / partly / no)

---

## Usage

### Basic workflow

1. **Describe a task** in the text input, e.g.:
   > "Build a sequence to pick up component A and place it on the goniometer"

2. The agent enters **planning phase**: it queries the assembly database, identifies required services, and builds a sequence.

3. When satisfied with the plan, say:
   > "Execute the sequence"

4. The agent enters **executing phase**: it runs actions one by one, adjusting parameters on failure and escalating to the planner for complex errors.

5. After execution, the agent optionally records what it learned about the services used.

### GUI controls

| Control | Function |
|---|---|
| **Listen** | Record microphone input → transcribed via Whisper → inserted into chat |
| **Record Knowledge** | After execution, the agent records service usage notes to the KB |
| **Consolidate Knowledge** | Every 3 executions, merge and clean the KB |
| **Edit → New Thread** | Clear conversation history, start fresh |
| **Edit → Update Assistant Files** | Reload service/frame files from disk |

### Sequence modification

You can ask the agent to modify an existing sequence in natural language:

> "Move the gripping step before the vision step"
> "Delete the last action"
> "Change the speed parameter of action 3 to 0.5"
> "Save this sequence as pick_and_place"
> "Load the sequence named goniometer_alignment"

### Error recovery

When an action fails during execution, the executor LLM attempts to fix parameters and retry. If it cannot resolve the error, the agent escalates to the planner, which diagnoses the issue and either repairs the sequence or asks the user for guidance.
