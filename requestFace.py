import requests

url = "http://localhost:5000/detect_face"
files = {
    "image": open("/home/dangaradhya/aps360/Projects/detected_face.jpg", "rb")
}
response = requests.post(url, files=files)

# Save or display the detected face image if the request is successful
if response.status_code == 200:
    with open("detected_face.jpg", "wb") as f:
        f.write(response.content)
else:
    print(response.text)
    print(response.json())
