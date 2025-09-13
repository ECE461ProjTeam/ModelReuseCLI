# A CLI for Trustworthy Pre-Trained Model Re-Use

## Setup
Put your Gemini-2.0-flash API key into gemini_key.txt

Optionally, set up a Hugging Face token for higher rate limits:
- Preferred: set an environment variable `HF_TOKEN`
- Or create a file `hf_key.txt` with your token on the first line (this file is ignored by git)
- To avoid OS permission issues with Hugging Face, set a project-local cache

## Usage
Run ```./run install``` to install dependencies or ```./run URL_FILE``` where URL_FILE is filepath to list of URLs

## Contributors
- Mikhail Golovenchits
- Vatsal Dudhaiya
- Murad Ibrahimov
- Jake Scherer