# A CLI for Trustworthy Pre-Trained Model Re-Use

## Setup
Put your Gemini-2.0-flash API key into gemini_key.txt

Optionally, set up a GitHub token for higher rate limits:
- Preferred: set an environment variable `GITHUB_TOKEN`
- Or create a file `git_token.txt` with your token in the first line

Optionally, set up a Hugging Face token for higher rate limits:
- Preferred: set an environment variable `HF_TOKEN`
- Or create a file `hf_key.txt` with your token on the first line (this file is ignored by git)
- To avoid OS permission issues with Hugging Face, set a project-local cache

## Usage
Run ```./run install``` to install dependencies or ```./run URL_FILE``` where URL_FILE is filepath to list of URLs

## Testing URL Parsing
To test the URL parsing functionality, create a text file with CSV format:
```
<code_link_1>, <dataset_link_1>,<model_link_1>
<code_link_2>, <dataset_link_2>,<model_link_2>
... and so on
```

Then run: ```./run your_test_file.txt```
Note: I just used the sample_input.txt from Piazza

## Contributors
- Mikhail Golovenchits
- Vatsal Dudhaiya
- Murad Ibrahimov
- Jake Scherer