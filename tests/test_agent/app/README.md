# Realtime Demo App

This folder now contains only the frontend demo assets used by the main Bodhi server.

## Usage

Run the project server from the repository root:

```bash
poetry run python -m bodhi.servicer.server
```

Then open your browser to: http://localhost:8000

## How to Use

1. Click **Connect** to establish a realtime session
2. Audio capture starts automatically - just speak naturally
3. Click the **Mic On/Off** button to mute/unmute your microphone
4. To send an image, enter an optional prompt and click **🖼️ Send Image** (select a file)
5. Watch the conversation unfold in the left pane (image thumbnails are shown)
6. Monitor raw events in the right pane (click to expand/collapse)
7. Click **Disconnect** when done

### Human-in-the-loop approvals

- The seat update tool now requires approval. When the agent wants to run it, the browser shows a `window.confirm` dialog so you can allow or deny the tool call before it executes.

## Architecture

- **Backend**: FastAPI WebSocket endpoint under `src/bodhi/api/endpoint.py`
- **Session Management**: per-connection realtime session via OpenAI SDK Realtime
- **Image Inputs**: chunked upload and server-side `conversation.item.create` forwarding
- **Audio Processing**: 24kHz mono audio capture and playback
- **Event Handling**: history sync, typed delta mapping, tool approval loop
- **Frontend**: vanilla JavaScript assets in this folder
