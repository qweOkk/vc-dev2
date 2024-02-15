import  json

with open("speakers_stats_test.json") as f:
    data = json.load(f)

spk_list = []
for wav_basename in data:
    spk_name = wav_basename.split('_')[0]
    spk_list.append(spk_name)

print(len(set(spk_list)))
