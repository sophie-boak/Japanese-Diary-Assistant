## Japanese Diary Assistant 
<div align="center">
手書きの日記から、AIのアドバイスまで
</div>

Just take a photo of your Japanese writing, let the program extract the text from, and get helpful tips on grammar and vocabulary to improve your skills.

# Reason Behind the App
To practice my Japanese writing skills, I would write a diary everyday and then type it into an AI chatbot to get feedback on my grammar, vocabulary and correct any mistakes. I wanted a quick way to get feedback without having to retype my journal every time while also benefitting from handwriting Japanese, especially kanji. So I created this PyQt App with a Studio Ghibli Totoro theme.

# Features
- You first take a photo of your diary entry using your computer's live camera and accept the photo that has been taken or retake the photo
<img width="477" height="313" alt="1 photo-taking" src="https://github.com/user-attachments/assets/23440448-d078-4533-9272-d66af366edef" />

<img width="477" height="313" alt="2 photo-preview" src="https://github.com/user-attachments/assets/807cb75f-845f-468b-811e-1c86a56122e1" />

- Next, the photo is sent to the Google Cloud Vision API which uses OCR (Optical Character Recognition) to extract the handwritten Japanese text from the image. The text is also editable in case the OCR could not recognise the handwritting. However, by using the document text detection of the API client, the results have been very accurate.

<img width="477" height="313" alt="3 OCR" src="https://github.com/user-attachments/assets/94100ff2-84ac-403d-a1f0-941d21a15205" />

- After sending the text to Gemini API for feedback, users are sent to a loading page while they wait for feedback. This is possible by using QThread so that the GUI does not freeze while waiting for the API response.

<img width="477" height="313" alt="4 loading-page" src="https://github.com/user-attachments/assets/7bd275b5-e84e-470a-8e93-34479e6da7a1" />

- Lastly, it returns feedback on your diary entry. This includes a correct version of your diary, a breakdown of the corrections and more detailed explanation on why as well as more suggestions on how to improve your entry such as new vocabulary or alternative ways to phrase a sentence.

<img width="477" height="313" alt="5 AI-advice" src="https://github.com/user-attachments/assets/1c16ab7b-4d16-4993-a057-afcf30e79b63" />

<img width="477" height="313" alt="6 more-AI-advice" src="https://github.com/user-attachments/assets/8dd90eb8-88b7-4b8f-ada3-6aa19c813fc7" />

<img width="477" height="313" alt="7 more-suggestions" src="https://github.com/user-attachments/assets/8d8c34b5-fd5a-49ba-bafd-fab7fe71ed05" />

# Technologies Used
- Python & PyQt5 - desktop GUI
- Google Cloud Vision API - OCR text extraction
- Google Gemini API - AI feedback
- OpenCV - camera feed

# Credits
- **Studio Ghibli** - original creators of Totoro and the visual inspiration for the app's theme
- **Fan artists** - Totoro-inspired artwork used in screenshots were sourced from social media. Original artists unknown. All artwork belongs to their respective creators and is not included in this repsitory. 
