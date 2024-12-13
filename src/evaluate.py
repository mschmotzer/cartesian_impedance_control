import json

# Load the JSON data
file_path = "robot_state_log_2024_12_11_1043.json"  # Replace with the path to your JSON file
entries = []

with open(file_path, 'r') as file:
    for line in file:
        entries.append(json.loads(line.strip()))

# Initialize variables
lost_count = 0
lost_start_time = None
recovery_times = []

# Process each entry
for entry in entries:
    success = entry.get("Tracking", {}).get("Success", False)
    current_time = entry.get("Time", {}).get("Time", 0.0)
    
    if not success:
        lost_count += 1
        if lost_count == 1:
            lost_start_time = current_time  # Record the first false time
    else:
        if lost_count > 2:  # Hand was "lost"
            if lost_start_time is not None:
                recovery_time = current_time - lost_start_time
                recovery_times.append(recovery_time)
        # Reset lost tracking
        lost_count = 0
        lost_start_time = None

# Calculate the average recovery time
if recovery_times:
    average_recovery_time = sum(recovery_times[:-1]) / (len(recovery_times))
    print(f"Average recovery time: {average_recovery_time:.2f} seconds")
else:
    print("No hand loss and recovery events detected.")
