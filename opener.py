import json

with open("file.JSON", "r") as file:
    data = json.load(file)
 
print(data['properties']['meta'])