#!/bin/bash

input_dir="./classes/no_fall"
output_dir="./classes/no_fall"

# Boucle à travers tous les fichiers .avi du dossier spécifié
for input_file in "$input_dir"/*.mp4; do
    # Obtenez le nom de base du fichier sans extension
    base_name=$(basename "$input_file" .mp4)
    output_file="$output_dir/${base_name}_conv.mp4"
    
    # Convertir en utilisant ffmpeg
    ffmpeg -i "$input_file" -vcodec libx264 "$output_file"
done