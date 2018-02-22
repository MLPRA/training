import json
import os

path = 'C:/Users/fried/Desktop/images_json'

files = os.listdir(path)
for file in files:
	file_name, file_extension = os.path.splitext(file)
	file_path = '{}/{}'.format(path, file)
	if file_extension.lower() == '.json':
		with open(file_path) as f:
			data = json.load(f)
		if (data['label'] == 'none'):
			data['label'] = 'other'
			with open(file_path, 'w') as out:
				json.dump(data, out, indent=2)
			print(data)
