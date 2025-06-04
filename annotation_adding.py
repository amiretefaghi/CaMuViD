import json

def load_coco_json(file_path):
    """Load the COCO format JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_next_annotation_id(data):
    """Find the largest annotation ID and return the next available ID."""
    if 'annotations' in data and data['annotations']:
        ids = [ann['id'] for ann in data['annotations']]
        max_id = max(ids)
        next_id = max_id + 1
    else:
        next_id = 1  # Start from 1 if no annotations present
    return next_id

def add_annotation(data, annotation):
    """Add a new annotation to the COCO JSON."""
    data['annotations'].append(annotation)

def save_coco_json(file_path, data):
    """Save the updated COCO JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    # Path to your JSON file
    file_path = '/blue/hmedeiros/amir.etefaghidar/workspace/DuViDA/data/Wildtrack/Wildtrack_coco_test_anno_chunk_id.json'

    # Load the JSON file
    coco_data = load_coco_json(file_path)

    # Continuously add new annotations until user quits
    while True:
        # Get the next available annotation ID
        next_id = get_next_annotation_id(coco_data)
        print(f"Next available annotation ID is: {next_id}")

        # Prompt for new annotation details
        print("Enter the annotation details in dictionary format (or type 'q' to quit):")
        annotation_input = input()

        # Exit loop if user inputs 'q'
        if annotation_input.lower() == 'q':
            break

        # Convert the input string to a dictionary
        try:
            new_annotation = eval(annotation_input)
            # new_annotation['id'] = next_id  # Assign the new ID

            # Add the new annotation
            add_annotation(coco_data, new_annotation)

            print(f"Annotation with ID {new_annotation['id']} added successfully.")
        except Exception as e:
            print(f"Error: {e}")
            continue  # Continue the loop even if an error occurs

    # Save the updated JSON
    save_coco_json(file_path, coco_data)
    print("All annotations saved successfully.")

if __name__ == "__main__":
    main()
