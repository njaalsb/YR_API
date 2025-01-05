# Script for å overvåke filendringer med et gitt intervall, deretter printe deler av innholdet i filen
# Langt intervall her ettersom værmeldingene ikke oppdateres spesielt ofte (1t intervall her)

import os
import time
import json

def detect_file_changes(file_path, interval = 3600):
    last_modified = os.path.getmtime(file_path)
    while True:
        current_modified = os.path.getmtime(file_path)
        if current_modified != last_modified:
            print('filendring')
            with open("file.JSON", "r") as file:
                data = json.load(file)

            print(data['properties']['timeseries'][0])
    
            last_modified = current_modified
        time.sleep(interval)    
    
detect_file_changes('/yrposjekt/file.json')