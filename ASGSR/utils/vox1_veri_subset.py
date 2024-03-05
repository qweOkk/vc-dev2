# There are too many data pairs in the vox1 verification set
# To much time of generating adversarial examples
# Number of verification speakers: 40
# Min speaker trails: id10301 168
# Max speaker trails: id10300 1040
import random


def random_sample_according_speaker():
    veri_file = open('/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/veri_test.txt', 'r')

    # calculate the number of speakers
    speakers = {}
    for line in veri_file:
        line = line.strip()
        line = line.split(' ')
        if line[0] == '0':
            if line[1].split('/')[0] not in speakers:
                speakers[line[1].split('/')[0]] = 1
            else:
                speakers[line[1].split('/')[0]] += 1
    for spk in speakers:
        print(spk, speakers[spk])

    # calculate each trials of speakers, save in spk_trials dic, key is speaker id, value is trials (list)
    veri_file = open('/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/veri_test.txt', 'r')
    veri_file_10 = open('/mntnfs/lee_data1/wangli/ASGSR/veri_test_10_0.txt', 'w')

    spk_trials = {}
    for line in veri_file:
        line = line.strip()
        if line.split()[0] == '0':
            spk_id = line.split()[1].split('/')[0]
            if spk_id not in spk_trials:
                spk_trials[spk_id] = [line]
            else:
                spk_trials[spk_id].append(line)

    # Then select 10% of the trials of each speaker
    # save in veri_test_10_0.txt
    for spk in spk_trials:
        trials = spk_trials[spk]
        trials = random.sample(trials, int(round(len(trials) * 0.1)))
        for trial in trials:
            veri_file_10.write(trial + '\n')
    veri_file.close()


def trial_uniform_sample(proportion=0.1):
    veri_file = open('/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/veri_test.txt', 'r')
    record_file = open('vox1_uniform_sample_{}.txt'.format(int(proportion * 100)), 'w')
    trial_dict = {}
    for line in veri_file.readlines():
        line = line.strip().split(' ')
        spk1 = line[1].split('/')[0]
        spk2 = line[2].split('/')[0]
        trial_id = '{}_{}'.format(spk1, spk2)
        if trial_id not in trial_dict:
            trial_dict[trial_id] = [line]
        else:
            trial_dict[trial_id].append(line)

    for trial_id in trial_dict:
        n = int(len(trial_dict[trial_id]) * proportion)
        if n == 0:
            n = 1
        trials = random.sample(trial_dict[trial_id], n)
        for trial in trials:
            print(trial)
            record_file.write(' '.join(trial) + '\n')

    veri_file.close()
    record_file.close()


if __name__ == '__main__':
    trial_uniform_sample(0.25)
