#!/bin/bash

# Parse the configuration file
config_file="config.json"
config=$(jq -r 'to_entries|map({key: .key, value: .value})|from_entries' "$config_file")

# Specify the files to transfer
folder_to_transfer="hermes"
folder_2="examples"

# Iterate through each worker in the configuration
for worker in $(jq -r 'keys[]' "$config_file"); do
    hostname=$(jq -r ".$worker.ip" "$config_file")
    username=$(jq -r ".$worker.username" "$config_file")
    password=$(jq -r ".$worker.password" "$config_file")
    path=$(jq -r ".$worker.path" "$config_file")

    # Transfer the JSON files to the parent directory

    # Transfer the folder
    sshpass -p "$password" scp -r "$folder_to_transfer" "$username@$hostname:$path"
    sshpass -p "$password" scp -r "$folder_2" "$username@$hostname:$path"

    echo "Files transferred to $hostname!"
    done

echo "Files transferred successfully!"