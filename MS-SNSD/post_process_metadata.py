import json

input_path = "/nvme/uniamphion/se/Libritts_SE/metadata.json"

with open(input_path, 'r') as f:
    data = json.load(f)

for i in range(len(data)):
    data[i]['Path'] = data[i]['Path'].replace('/nvme/uniamphion/se/Libritts_SE/', '')
    for j in range(len(data[i]['NoisePath'])):
        data[i]['NoisePath'][j] = data[i]['NoisePath'][j].replace('/nvme/uniamphion/se/Libritts_SE/', '')
    for j in range(len(data[i]['NoisySpeechPath'])):
        data[i]['NoisySpeechPath'][j] = data[i]['NoisySpeechPath'][j].replace('/nvme/uniamphion/se/Libritts_SE/', '')

output_path = '/nvme/uniamphion/se/Libritts_SE/metadata_updated.json'
with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)
