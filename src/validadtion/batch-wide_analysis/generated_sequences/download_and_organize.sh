#!/bin/bash

# Function to download files from Google Drive (ok, this actually doesnt work because our drive is private, but you get the idea)
download_files() {
    gdown --folder "https://drive.google.com/drive/folders/1PMl19UFs-aAys5pAAyiRSUbTAtIiaFx9?usp=sharing"
}

# Function to extract date from file name
extract_date() {
    file_name="$1"
    if [[ $file_name =~ ([0-9]{4}-[0-9]{2}-[0-9]{2}) ]]; then # YYYY-MM-DD
        echo "${BASH_REMATCH[1]}"
    elif [[ $file_name =~ ([a-zA-Z]{3}[0-9]{2}) ]]; then # MonDD
        echo "${BASH_REMATCH[1]}"
    else
        echo "unknown_date"
    fi
}

# Main script
download_files

for file in *; do
    if [[ -f $file ]]; then
        date=$(extract_date "$file")
        folder_name="sequences_$date"
        mkdir -p "$folder_name"
        mv "$file" "$folder_name/"
    fi
done
