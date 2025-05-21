<p align="center">
  <img src="resources/system/uniframe_logo_2.png" alt="Uniframe Studio" width="200"/>
</p>

<h1 align="center" style="color:#ebf0d2; font-family: monospace;">Uniframe Studio</h1>
<p align="center" style="color:#aaaaaa;">
  An AI-powered dubbing system — transcribe, translate, synthesize, align.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/License-Apache 2.0-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Beta-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/TTS-ElevenLabs%20%7C%20OpenAI-darkgreen?style=flat-square" />
</p>


## Overview

**Uniframe Studio** is an intelligent dubbing system designed to automate the process of video translation and voiceover with high temporal precision. It synthesizes a new audio track using TTS models and precisely aligns the original video to match the timing of the new voiceover.

---

## 🎬 Demo

_Coming soon..._

---

## 🔄 Workflow

### Audio Pipeline

1. **Extract audio** from video.
2. **Transcribe** with both segment-level and word-level timestamps to support the system’s meaning-aware translation process.
3. **Transcript correction** (prepare transcription segments for translation).
4. **Translate** each corrected segment into the target language.
5. **Voiceover** generation using TTS:
   - ElevenLabs (including original voice cloning)
   - OpenAI (multiple voice options)
6. **Audio synthesis**: Build the final audio track (mono and stereo versions).

### Video Alignment

1. **Split original video into segments** using transcript timestamp data.
2. **Stretch/compress video segments** to match new voiceover segment durations.
3. **Merge adjusted segments** into a final video synced with the new audio track.

---

## Features

- REST API for easy integration with other systems.
- CLI interface with individual commands for each processing step.
- Accurate transcription and it's structure correction.
- Multi-language support via high-quality translation.
- Voiceover with choice of synthetic voices.
- Auto-assembled audio tracks (mono/stereo).
- Frame-accurate video reassembly aligned with new speech duration.

---

# API Usage
The system provides a REST API for easy integration:

_Coming soon..._

---

## CLI Usage

All processing steps are available individually through the CLI interface:

_Coming soon..._

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 🤗

---

## ⚖️ License

This project is licensed under the Apache 2.0 License.
