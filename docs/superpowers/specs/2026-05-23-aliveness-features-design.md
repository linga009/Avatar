# Avatar v3.10 Aliveness Features Design Spec

**Date:** 2026-05-23
**Version:** Avatar v3.10

## Goal

Four features to make Avatar more alive: speech recognition (understand spoken words), proactive notifications (initiate communication), topic diversity (broader exploration), and neural TTS (natural voice).

## Feature 1: Whisper Tiny Speech Recognition

`openai/whisper-tiny` (39M params, ~150MB CPU RAM) transcribes microphone audio when speech is detected.

- New file: `halo3/senses/speech_recognition.py` — `SpeechRecognizer` class
- Uses `faster-whisper` (CTranslate2 backend) for speed on CPU
- Called in `main.py` only when `sensory_stats.speech_detected == True`
- Transcribed text → PFC context as `heard_speech="..."`
- Transcribed text → `organism.tick()` as `heard_speech` param
- Avatar references heard speech in meta-thoughts and chat
- Dockerfile: `pip install faster-whisper`
- CPU only, ~500ms per 2s chunk, only when speech detected

## Feature 2: Proactive Notifications

Avatar pushes messages when significant events happen.

Triggers:
- Discovery (r > 0.6 + finding text)
- Meditation insight
- GWT ignition after dark > 10 ticks
- Self-surprise > 0.5

- `chat_server.py`: `_proactive_messages` deque + `GET /notifications` endpoint
- Chat UI: polls `/notifications` every 30s, displays as Avatar-initiated
- `organism.tick()` returns `proactive_message` when events fire
- `main.py` pushes to chat server queue

## Feature 3: Topic Diversity Enforcement

- `organism.py`: auto-saturation threshold 50 → 20 ticks (same topic, no r gain)
- `volatility.py`: recency penalty — topics explored in last 30 ticks get 0.7x BS value

## Feature 4: Piper Neural TTS

- `tts_narration.py`: add `_narrate_piper()` alongside espeak
- Config: `tts_mode: str = "piper"` (default changed from espeak)
- Piper binary + en_US voice installed in Dockerfile
- Falls back to espeak if Piper unavailable
- Same output format: 16kHz, 2s, float32

## Resource Budget

| Feature | VRAM | CPU RAM | Per-tick |
|---------|------|---------|----------|
| Whisper tiny | 0 | ~150 MB | ~500ms (speech only) |
| Proactive notifications | 0 | 0 | 0 |
| Topic diversity | 0 | 0 | 0 |
| Piper TTS | 0 | ~65 MB | ~300ms (every 3rd tick) |
| **Total** | **0** | **~215 MB** | fits in 12GB WSL2 |

## Files Changed

| File | Change |
|------|--------|
| `halo3/senses/speech_recognition.py` | New — SpeechRecognizer class |
| `halo3/chat_server.py` | Proactive messages queue + /notifications endpoint |
| `halo3/psyche/organism.py` | Proactive message generation + auto-saturation 50→20 + heard_speech param |
| `halo3/psyche/volatility.py` | Recency penalty on BS valuation |
| `halo3/senses/tts_narration.py` | Piper TTS method |
| `halo3/config.py` | tts_mode default espeak→piper |
| `halo3/main.py` | Wire speech recognition + proactive messages |
| `Dockerfile` | Install faster-whisper + piper-tts |
