import google.generativeai as genai

genai.configure(api_key="AIzaSyA96OScR17VpKckBCr-NHDm7J8-v0Ed6Uc")  # Replace with your actual API key

model = genai.GenerativeModel('gemini-2.0-flash')  # Or 'gemini-pro-vision' for multimodal

query = input("Enter your search: ")
response = model.generate_content(query)

print(response.text)