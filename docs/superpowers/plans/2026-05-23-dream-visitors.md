# Dream Visitors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Whisper and Kokoro as dream-only teachers — generate enriched (audio, text) pairs during sleep, train Avatar's FNO + contrastive alignment on GPU in a new Phase 5 subprocess.

**Architecture:** Phase 5a (CPU: Whisper transcribes audio archive), Phase 5b (CPU: Kokoro narrates discoveries), Phase 5c (GPU subprocess: train FNO on enriched pairs). Models load/teach/unload — never present during waking.

**Tech Stack:** faster-whisper, kokoro-onnx, JAX/Equinox, existing CLion optimizer

---

### Task 1: Audio archive in SenseBuffer
### Task 2: Dream visitors CPU pair generation (dream_visitors.py)
### Task 3: Dream visitors GPU worker (dream_visitors_worker.py)
### Task 4: Wire Phase 5 into main.py dream cycle
### Task 5: Config params + Dockerfile
### Task 6: Rebuild and verify
