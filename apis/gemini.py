import requests
import json

def prompt_gemini(prompt, api_key):
    """
    Make a request to Google's Gemini API to generate responses based on a text prompt.
    
    Args:
        prompt (str): The text prompt to send to the model
        api_key (str): Gemini API key
    
    Returns:
        generated_text (str): Gemini's response
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': api_key
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        generated_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        
        return generated_text
    
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None

if __name__ == "__main__":
    with open('gemini_key.txt', 'r') as file:
        GEMINI_API_KEY = file.readline().strip()
    
    prompt_text = "How do you say Hi in French?"
    
    result = prompt_gemini(prompt_text, GEMINI_API_KEY)
    
    if result:
        try:
            print(f"\nGenerated Response: {result}")
        except (KeyError, IndexError) as e:
            print(f"Error extracting response: {e}")
    else:
        print("Request failed")