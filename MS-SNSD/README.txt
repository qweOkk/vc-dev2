=======Dataset Name======= 
Libritts_SE

=======General Description========
Libritts_SE is a self-created clean and noisy parallel speech database designed to train and test speech enhancement methods sampled at 16kHz. The dataset contains a train set and a test set. The train set contains 500 hours of noisy speech, 500 hours of noise files that are used to create the noisy speech, and 100 hours of clean speech from LibriTTS-train-clean-360. Each clean speech has 5 corresponding noisy speech at 5 different signal-to-noise (SNR) levels. The test set contains 50 hours of noisy speech, 50 hours of noise files that are used to create the noisy speech, and 10 hours of clean speech from LibriTTS-test-clean. Similar to the train set, each clean speech in the test set has 5 corresponding noisy speech at 5 different signal-to-noise levels.


=======Dataset Creation Detail=========
The dataset creation method follows "A scalable noisy speech dataset and online subjective test framework," in Interspeech, 2019, with source code available at https://github.com/microsoft/MS-SNSD/tree/master. 2 main modifications are made to the original paper's method: 1) While the original paper uses clean speech from VoiceBank+DEMAND as the clean speech corpus, this work uses LibriTTS-train-clean-360 and LibriTTS-test-clean instead so as to increase the speech content and speaker diversity 2) While the original paper only provides 12 noise types, this work introduces 14 additional noise types to the noise corpus so as to increase the noise diversity.

Clean speech dataset creation detail (CleanSpeech_training, CleanSpeech_testing)
	- Each clean speech clip is at least 10 seconds
	- Each clean speech clip is randomly sampled from train-clean-360 of libritts for training and from test-clean of libritts for testing. A random seed of 42 is used, and for this random seed, 886 speakers are picked for creating the training set and 38 speakers are picked for creating the testing set.
	- If a clean speech clip sampled from libritts is less than 10 seconds, additional speech clips will be randomly sampled from libritts and concatenated together until the overall length of the concatenated clean speech clip is longer than 10 seconds. When concatenating multiple speech clips, a 0.2 second of silence is introduced between each speech clip.
	- The total duration of the clean speech dataset created for training is 100 hours, with a total of 24,977 utterances
	- The total duration of the clean speech dataset created for testing is 10 hours, with a total of 2,464 utterances

Noise speech dataset creation detail (Noise_training, Noise_testing)
	- For each clean speech clip, a noise speech clip of the same duration is created.
	- Each noise clip is randomly sampled from a pool of noise clips
		- Pool of noise clips
			- The noise clips are selected from DEMAND database and Freesound.org
			- Each noise clip is hand-picked by carefully listening to ensure the quality of the recordings
			- There are 26 noise types in total, with 21 for training and 5 for testing. The numbers in () are the number of audio files picked for each noise type:
				Air conditioner (10)
				Babble (12)
				copy machine (9)
				door shutting (10)
				eating (munching) (10)
				neighbour speaking (14)
				squeaky chair (11)
				vacuum cleaner (9)
				car noise (10)
				cafe noise (10)
				Office (9)
				Construction (10)
				Siren (9)
				Metro (8)
				Park (12)
				Rain (9)
				Mall (10)
				Footstep (8)
				Dog (10)
				Seawaves (9)
				WasherDryer (7)
				Street (10) [for testing]
				Kitchen (8) [for testing]
				AirportAnnouncement (11)  [for testing]
				Wind (9) [for testing]
				Typing (10) [for testing]
	- If a noise clip sampled is less than the duration of the clean speech, additional noise clips from the same noise type will be randomly sampled and concatenated together until the overall length of the concatenated noise speech clip is equivalent to the clean speech. When concatenating multiple noise clips, a 0.2 second of silence is introduced between each noise clip.
	- The total duration of the noise dataset created for training is 500 hours, with a total of 124,885 utterances
	- The total duration of the noise dataset created for testing is 50 hours, with a total of 12,320 utterances

Noisy speech dataset creation detail (NoisySpeech_training, NoisySpeech_testing)
	- For each clean speech clip, 5 noisy speech clips are created
		- Each of the 5 noisy speech clips has a different SNR value of
			- 0, 10, 20, 30, 40 respectively for training
			- 2, 12, 22, 32, 42 respectively for testing
		- The 5 noisy speech clips are created using the same noise clip
	- The total duration of the noisy speech dataset created for training is 500 hours, with a total of 124,885 utterances
	- The total duration of the noisy speech dataset created for testing is 50 hours, with a total of 12,320 utterances