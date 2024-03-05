'''
train_file = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt'
dev_file = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt'
eval_file = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt'

number of speakers: 107
number of audio per speaker: 270

107 x 106 = 11342
'''
import random


def generate_bonafide_list():
    train_file = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt'
    dev_file = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt'
    eval_file = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt'
    file_list = []
    for file in [train_file, dev_file, eval_file]:
        with open(file, 'r') as f:
            for line in f.readlines():
                if 'bonafide' in line:
                    file_list.append(line.strip())
    return file_list


def generate_speaker_dic(file_list):
    '''
    Args:
        file_list: a list of files, format of ASVspoof2019,
             eg: ['PA_0096 PA_T_0005376 ccc - bonafide', 'PA_0098 PA_T_0005397 ccc - bonafide']


    Returns:
        dicionary, key: speaker id, value: a list of AUDIO_FILE_NAME

    '''
    speaker_dic = {}
    for file in file_list:
        speaker_id = file.split(' ')[0]
        if speaker_id not in speaker_dic:
            speaker_dic[speaker_id] = []
        speaker_dic[speaker_id].append(file.split(' ')[1])

    return speaker_dic


def sample_enroll_eval_pairs(speaker_dic):
    '''
    Args:
        speaker_dic: a dictionary, key: speaker id, value: a list of AUDIO_FILE_NAME
        sample enroll and eval pairs
        all pairs are diff speaker
        all speaker sample one eval audio in each other speaker
        Sample pairs that need to be excluded from swapping: (PA_T_0005387 PA_T_0005353) (PA_T_0005353 PA_T_0005387)

    Returns:
        enrollment evaluation pairs, write to txt file
    '''
    result_pairs = []

    # diffenent speaker
    for enroll_speaker_id, file_ids in speaker_dic.items():
        other_speakers = list(speaker_dic.keys())
        other_speakers.remove(enroll_speaker_id)
        for eval_speaker_id in other_speakers:
            while True:
                enroll_file_id = random.choice(file_ids)
                eval_file_id = random.choice(speaker_dic[eval_speaker_id])

                enroll_eval_pair = (enroll_speaker_id, enroll_file_id, eval_speaker_id, eval_file_id, '0')
                eval_enroll_pair = (eval_speaker_id, eval_file_id, enroll_speaker_id, enroll_file_id, '0')

                if enroll_eval_pair not in result_pairs and eval_enroll_pair not in result_pairs:
                    result_pairs.append(enroll_eval_pair)
                    break

    # same speaker
    for speaker_id, file_ids in speaker_dic.items():
        for i in range(106):
            while True:
                enroll_file_id = random.choice(file_ids)
                eval_file_id = random.choice(file_ids)

                if enroll_file_id == eval_file_id:
                    continue

                enroll_eval_pair = (speaker_id, enroll_file_id, speaker_id, eval_file_id, '1')
                eval_enroll_pair = (speaker_id, eval_file_id, speaker_id, enroll_file_id, '1')

                if enroll_eval_pair not in result_pairs and eval_enroll_pair not in result_pairs:
                    result_pairs.append(enroll_eval_pair)
                    break

    return result_pairs


if __name__ == '__main__':
    file_list = generate_bonafide_list()
    speaker_dic = generate_speaker_dic(file_list)
    sample_enroll_eval_pairs(speaker_dic)
    enroll_eval_pairs = sample_enroll_eval_pairs(speaker_dic)
    with open('temp.txt', 'w') as f:
        for pair in enroll_eval_pairs:
            print(pair)
            f.write(' '.join(pair) + '\n')
