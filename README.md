# Graph-Based Audio Encryption — Code and Voice Samples

This repository contains only the research Python source code and four audio recordings used for the audio encryption paper. It contains **no results, figures, metrics, or reports**.

- `code/original_research_script.py`: original supplied code, including the original local input path.
- `code/run_experiment.py`: command-line runnable adaptation for different audio files. It prints experiment metrics when executed but does not include precomputed results.
- `voices/original/`: original submitted audio files, including MP3-format speech.
- `voices/wav_16bit/`: 16-bit PCM WAV versions compatible with the Python code.

Install with `pip install -r requirements.txt`.

Example run from the repository folder:

```bash
python code/run_experiment.py --audio voices/wav_16bit/bird.wav --key-trials 1
```

The script uses fixed demonstration key/nonce parameters for reproducible research and is not security-audited. Do not use it for real confidential audio. Verify redistribution permission for all recordings before publishing them on GitHub.
