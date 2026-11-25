# AI Desktop Companion and Autonomous Voice Agent

A real-time, multimodal voice and text AI desktop companion implemented in Python. By integrating low-latency speech-to-text, expressive text-to-speech, a two-dimensional valence-arousal mood engine, persistent vector memory, visual context awareness (screen and webcam), and autonomous graphical user interface (GUI) control, this system facilitates natural conversational interaction while executing tasks directly on the host machine.

## Key Features

* **Real-Time Voice and Text Interaction**
  * **Voice Activity Detection (VAD) and Transcription:** Utilizes `faster-whisper` (`large-v3-turbo`) with pre-roll audio buffering to prevent clipping of initial syllables.
  * **Vocal Delivery Analysis:** Evaluates microphone root-mean-square (RMS) energy and speaking pace (words per second) via `librosa` to supply contextual tone indicators to the language model.
  * **Multi-Provider Text-to-Speech (TTS):** Supports **Fish Audio** (including inline expression tags), **ElevenLabs**, **Chatterbox Turbo** (local GPU voice synthesis), and **Alibaba DashScope**.

* **Persistent Semantic Memory (`chroma.py`)**
  * Built on **ChromaDB** and `Qwen/Qwen3-Embedding-0.6B`.
  * Summarizes idle conversation segments into categorized records (`[user]`, `[assistant_answer]`, `[assistant_advice]`, `[agreed_plan]`) featuring automated deduplication and recency-weighted semantic retrieval.

* **Dynamic Two-Dimensional Mood Engine (`mood.py`)**
  * Models emotional states within a unit circle governed by **Valence** ($x \in [-1, 1]$) and **Arousal** ($y \in [-1, 1]$), incorporating configurable inertia and gradual decay toward a neutral baseline.
  * Dynamically adjusts system prompt tone directives and TTS expression tags according to the calculated mood vector angle and magnitude.

* **Multimodal Vision (Screen and Webcam)**
  * **Contextual Triggers:** Captures the primary display (`mss`) or webcam feed (`opencv-python`) when predefined visual trigger phrases are detected in user input.
  * **Idle Observations:** Periodically evaluates screen and camera context during extended periods of inactivity to initiate contextually relevant interactions.

* **Autonomous Desktop Control (`ai_hands.py`)**
  * A background vision-language model (VLM) worker loop that translates natural language directives into sequential GUI operations (`click`, `double_click`, `right_click`, `drag`, `type`, `key`, `scroll`, `wait`).
  * Incorporates visual feedback overlays (indicating current cursor coordinates and prior click locations), multi-step visual history, Windows DPI awareness, and `pyautogui` corner-failsafe termination.

* **Configurable Persona Framework**
  * Includes multiple predefined system personas: **Leo**, **Verity**, **Rocky**, **Teto**, **Cryptek**, **Tsundere**, **Mommy**, **Nova**, and **Sam**.

## System Architecture

| Module | Responsibility |
| :--- | :--- |
| `ai.py` | Core orchestrator managing concurrent audio/text threads, OpenRouter LLM streaming, structured JSON schema validation, visual triggers, TTS playback queues, and persona definitions. |
| `ai_hands.py` | Autonomous desktop control module utilizing `mss`, `pyautogui`, and an iterative visual feedback loop to perform multi-step GUI operations. |
| `chroma.py` | `ChatbotMemory` interface wrapping ChromaDB for persistent storage of timestamped conversation summaries and recency-blended vector queries. |
| `mood.py` | `MoodCircle` state machine managing emotional valence and arousal transitions and generating corresponding prompt instructions. |

## Prerequisites and Installation

### 1. System Requirements

* **Operating System:** Windows (recommended for full `ctypes` DPI awareness and `ai_hands.py` desktop automation) or Linux/macOS.
* **Hardware:** An NVIDIA GPU with CUDA support is strongly recommended for local `faster-whisper`, `Qwen3-Embedding-0.6B`, and `Chatterbox` TTS inference.
* **Runtime:** Python 3.10 or higher.

### 2. Dependency Installation

Clone the repository and install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Note: Ensure that a CUDA-compatible build of PyTorch is installed if executing Whisper, Chroma embeddings, or Chatterbox TTS locally on a GPU.*

### 3. Environment Configuration

Create a `.env` file in the root directory alongside `ai.py` and populate the required credentials for your selected services:

```env
# Required: OpenRouter API Key for LLM and Vision inference
OPENROUTER_API_KEY=your_openrouter_api_key

# TTS Provider Credentials (configure according to the active provider)
FISH_API_KEY=your_fish_audio_api_key

ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id

DASHSCOPE_API_KEY=your_alibaba_dashscope_key
ALIBABA_VOICE_ID=your_enrolled_voice_id
ALIBABA_REGION=ap-southeast-1
```

## Usage

1. **Launch the Application:**

   ```bash
   python ai.py
   ```

   *Note: By default, `ai.py` invokes `Assistant().run("Leo", "fish")` at the end of the file. Modify this call to `Assistant().run()` to select a persona and TTS provider interactively via the command line upon startup.*

2. **Interaction Protocol:**
   * Address the assistant via microphone or terminal input by including the active persona's name in the message (e.g., *"Leo, open YouTube"*).
   * Enter `q`, `quit`, or `exit` (or send a keyboard interrupt via `Ctrl+C`) to flush remaining conversation logs to ChromaDB and terminate the application.

3. **Desktop Automation Failsafe:**
   * While the GUI automation worker (`ai_hands.py`) is active, moving the mouse cursor rapidly into any corner of the primary screen will trigger `pyautogui.FAILSAFE` and immediately halt execution.

## Maintenance Note (`chroma.py`)

The bottom of `chroma.py` currently executes a database record deletion routine (`mem.collection.delete(ids=delete_ids)`) at the module level. Before deploying `ai.py`, ensure these lines are removed or encapsulated within an `if __name__ == "__main__":` block to prevent unintended execution upon import.

## License

This project is distributed under the terms of the [MIT License](LICENSE).