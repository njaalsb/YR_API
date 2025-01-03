#!/bin/bash

CURL='/usr/bin/curl'
a="Testbruker1app"
RVMHTTP="https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=51.5&lon=0"
CURLARGS="-A a -s"

# Respons lagres i variabel
raw="$($CURL $CURLARGS $RVMHTTP)"

# Variabelem lagres i log fil

echo "$raw" > file.json

# Printer innhold i denne filen til terminalen
cat file.json