import requests

url = "http://localhost:5001/predict"
files = {
    "image": open(
        "/home/dangaradhya/aps360/Projects/Classifier_CNN_Model/detected_face.jpg", "rb"
    )
}
response = requests.post(url, files=files)
print(response.json())
