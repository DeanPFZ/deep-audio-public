import unittest
import context
import json
import preprocess
import pandas as pd
from unittest import mock
from random import randint

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

h_classes = ['Human & Animal', 'Interacting Materials']
mapping = {'dog': 0,'rooster': 0,'pig': 0,'cow': 0,'frog': 0,'cat': 0,'hen': 0,
            'insects': 0,'sheep': 0,'crow': 0,'rain': 1,'sea_waves': 1,'crackling_fire': 1,
            'crickets': 0,'chirping_birds': 0,'water_drops': 1,'wind': 1,'pouring_water': 1,
            'toilet_flush': 1,'thunderstorm': 1,'crying_baby': 0,'sneezing': 0,'clapping': 0,
            'breathing': 0,'coughing': 0,'footsteps': 1,'laughing': 0,'brushing_teeth': 1,
            'snoring': 0,'drinking_sipping': 1,'door_wood_knock': 1,'mouse_click': 1,
            'keyboard_typing': 1,'door_wood_creaks': 1,'can_opening': 1,'washing_machine': 1,
            'vacuum_cleaner': 1,'clock_alarm': 1,'clock_tick': 1,'glass_breaking':1,'helicopter': 1,
            'chainsaw': 1,'siren': 1,'car_horn': 1,'engine': 1,'train': 1,'church_bells': 1,
            'airplane': 1,'fireworks': 1,'hand_saw': 1,
            }

# Test utilities that should be simple to finish
class TestUtilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._preprocessor = preprocess.Audio_Processor('')

    def test_set_audio_dir(self):
        self._preprocessor.set_audio_dir('hello')
        self.assertEqual(self._preprocessor._audio_dir, 'hello')

    def test_set_audio_sample_rate(self):
        self._preprocessor.set_audio_sample_rate(22050)
        self.assertEqual(self._preprocessor._sr, 22050)

class TestLoadAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._preprocessor = preprocess.Audio_Processor('../ESC-50/audio/')
        path_to_db='../ESC-50/'
        cls._dataset = pd.read_csv(path_to_db + 'meta/esc50.csv')
        classes = [None] * 50
        cls._dataset['h_category'] = None
        for index, row in cls._dataset.iterrows():
            target = row['target']
            classes[target] = row['category']
            cls._dataset.loc[index, 'h_category'] = mapping[row['category']]

    def test_load_fld_audio_wo_blocksize(self):
        samples, h_category, l_category = self._preprocessor._Audio_Processor__load_audio(
                                                                data=self._dataset[:10],
                                                                fld=1, blocksize=0, overlap=0)
        self.assertEqual(samples.shape, (10, 220500))
        self.assertEqual(len(h_category), 10)
        self.assertEqual(len(l_category), 10)

    def test_load_fld_audio_w_blocksize_only(self):
        samples, h_category, l_category = self._preprocessor._Audio_Processor__load_audio(
                                                                data=self._dataset[:10],
                                                                fld=1, blocksize=44100, overlap=0)
        self.assertEqual(samples.shape, (41, 1, 44100))
        self.assertEqual(len(h_category), 41)
        self.assertEqual(len(l_category), 41)

    def test_load_fld_audio_w_blocksize_w_overlap(self):
        samples, h_category, l_category = self._preprocessor._Audio_Processor__load_audio(
                                                                data=self._dataset[:10],
                                                                fld=1, blocksize=44100, overlap=4096)
        self.assertEqual(samples.shape, (49, 1, 44100))
        self.assertEqual(len(h_category), 49)
        self.assertEqual(len(l_category), 49)

    def test_load_audio_wo_blocksize(self):
        random = randint(0,50)
        samples, h_category, l_category = self._preprocessor._Audio_Processor__load_audio(
                                                                data=self._dataset[random:random+10],
                                                                fld=None, blocksize=0, overlap=0)
        self.assertEqual(samples.shape, (10, 220500))
        self.assertEqual(len(h_category), 10)
        self.assertEqual(len(l_category), 10)

    def test_load_audio_wo_blocksize_one_file(self):
        random = randint(0,50)
        samples, h_category, l_category = self._preprocessor._Audio_Processor__load_audio(
                                                                data=self._dataset[0:1],
                                                                fld=None, blocksize=0, overlap=0)
        self.assertEqual(self._dataset[0:1].shape[0], 1)
        self.assertEqual(samples.shape, (1, 220500))
        self.assertEqual(len(h_category), 1)
        self.assertEqual(len(l_category), 1)

    def test_load_audio_blocksize_one_file(self):
        random = randint(0,50)
        samples, h_category, l_category = self._preprocessor._Audio_Processor__load_audio(
                                                                data=self._dataset[5:6],
                                                                fld=None, blocksize=44100, overlap=0)
        self.assertEqual(self._dataset[0:1].shape[0], 1)
        self.assertEqual(samples.shape, (5, 1, 44100))
        self.assertEqual(len(h_category), 5)
        self.assertEqual(len(l_category), 5)

    def test_load_audio_blocksize_two_file(self):
        random = randint(0,50)
        samples, h_category, l_category = self._preprocessor._Audio_Processor__load_audio(
                                                                data=self._dataset[4:6],
                                                                fld=None, blocksize=44100, overlap=0)
        self.assertEqual(self._dataset[0:1].shape[0], 1)
        self.assertEqual(samples.shape, (10, 1, 44100))
        self.assertEqual(len(h_category), 10)
        self.assertEqual(len(l_category), 10)

    def test_load_audio_w_blocksize_only(self):
        random = randint(0,50)
        samples, h_category, l_category = self._preprocessor._Audio_Processor__load_audio(
                                                                data=self._dataset[random:random+10],
                                                                fld=None, blocksize=55125, overlap=0)
        self.assertEqual(samples.shape[1:], (1, 55125))

    def test_load_audio_w_blocksize_w_overlap(self):
        random = randint(0,50)
        samples, h_category, l_category = self._preprocessor._Audio_Processor__load_audio(
                                                                data=self._dataset[random:random+10],
                                                                fld=None, blocksize=55125, overlap=4096)
        self.assertEqual(samples.shape[1:], (1, 55125))

    def test_load_audio_w_multiple_fld_w_range(self):
        samples, h_category, l_category = self._preprocessor._Audio_Processor__load_audio(
                                                                data=self._dataset[25:40],
                                                                fld=range(1,5), blocksize=55125, overlap=0)
        self.assertEqual(samples.shape[1:], (1, 55125))

    def test_load_audio_w_multiple_fld_w_list(self):
        samples, h_category, l_category = self._preprocessor._Audio_Processor__load_audio(
                                                                data=self._dataset[25:40],
                                                                fld=[1,2,3,4], blocksize=55125, overlap=0)
        self.assertEqual(samples.shape[1:], (1, 55125))

class TestSpecGet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._preprocessor = preprocess.Audio_Processor('../ESC-50/audio/')
        path_to_db='../ESC-50/'
        cls._dataset = pd.read_csv(path_to_db + 'meta/esc50.csv')
        classes = [None] * 50
        cls._dataset['h_category'] = None
        for index, row in cls._dataset.iterrows():
            target = row['target']
            classes[target] = row['category']
            cls._dataset.loc[index, 'h_category'] = mapping[row['category']]

    def test_get_spec_model(self):
        print()
        model = self._preprocessor._Audio_Processor__spec_model((1,22500), True)
        self._preprocessor._Audio_Processor__check_model(model, True)
        print()

    def test_eval_model(self):
        cdata, targets, c_targets = self._preprocessor._Audio_Processor__load_audio(
                                                        data=self._dataset[0:10],
                                                        fld=None, blocksize=55125, overlap=4096)
        model = self._preprocessor._Audio_Processor__spec_model((1,55125), True)
        self._preprocessor._Audio_Processor__check_model(model)
        spec = self._preprocessor._Audio_Processor__evaluate_model(model, cdata)
        self.assertEqual(spec.shape, (42, 257, 216))
        

class TestMelGet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._preprocessor = preprocess.Audio_Processor('../ESC-50/audio/')
        path_to_db='../ESC-50/'
        cls._dataset = pd.read_csv(path_to_db + 'meta/esc50.csv')
        classes = [None] * 50
        cls._dataset['h_category'] = None
        for index, row in cls._dataset.iterrows():
            target = row['target']
            classes[target] = row['category']
            cls._dataset.loc[index, 'h_category'] = mapping[row['category']]

    def test_get_mel_spec_model(self):
        print()
        model = self._preprocessor._Audio_Processor__mel_spec_model((1,22500), 13, True, True)
        self._preprocessor._Audio_Processor__check_model(model, True)
        print()

    def test_eval_model(self):
        cdata, targets, c_targets = self._preprocessor._Audio_Processor__load_audio(
                                                        data=self._dataset[0:10],
                                                        fld=None, blocksize=55125, overlap=4096)
        model = self._preprocessor._Audio_Processor__mel_spec_model((1,55125), 13, True, True)
        self._preprocessor._Audio_Processor__check_model(model)
        spec = self._preprocessor._Audio_Processor__evaluate_model(model, cdata)
        self.assertEqual(spec.shape, (42, 13, 216))


class TestMFCCGet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._preprocessor = preprocess.Audio_Processor('../ESC-50/audio/')
        path_to_db='../ESC-50/'
        cls._dataset = pd.read_csv(path_to_db + 'meta/esc50.csv')
        classes = [None] * 50
        cls._dataset['h_category'] = None
        for index, row in cls._dataset.iterrows():
            target = row['target']
            classes[target] = row['category']
            cls._dataset.loc[index, 'h_category'] = mapping[row['category']]


    def test_extract_mfcc(self):
        data = self._preprocessor._Audio_Processor__preprocess_df(self._dataset[0:10],
                                                                kind='mfcc',
                                                                fld=None,
                                                                blocksize=55125,
                                                                overlap=4096,
                                                                n_mels=128,
                                                                power_melgram=True,
                                                                decibel_gram=True)
        self.assertEqual(data.shape[0], 42)
        self.assertEqual(data.shape[1], 116)


class TestWavenetGet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._preprocessor = preprocess.Audio_Processor('../ESC-50/audio/')
        path_to_db='../ESC-50/'
        cls._dataset = pd.read_csv(path_to_db + 'meta/esc50.csv')
        classes = [None] * 50
        cls._dataset['h_category'] = None
        for index, row in cls._dataset.iterrows():
            target = row['target']
            classes[target] = row['category']
            cls._dataset.loc[index, 'h_category'] = mapping[row['category']]

    def test_wavenet_no_blocksize(self):
        data = self._preprocessor._Audio_Processor__preprocess_df(self._dataset[0:1],
                                                                kind='wavnet',
                                                                fld=None,
                                                                blocksize=None,
                                                                overlap=None,
                                                                n_mels=None,
                                                                power_melgram=None,
                                                                decibel_gram=None)
        self.assertEqual(data.shape, (1, 16*3 + 2))

    def test_wavenet_no_blocksize_multiple(self):
        data = self._preprocessor._Audio_Processor__preprocess_df(self._dataset[0:2],
                                                                kind='wavnet',
                                                                fld=None,
                                                                blocksize=None,
                                                                overlap=None,
                                                                n_mels=None,
                                                                power_melgram=None,
                                                                decibel_gram=None)

        self.assertEqual(data.shape, (2, 16*3 + 2))

    def test_wavenet_blocksize(self):
        data = self._preprocessor._Audio_Processor__preprocess_df(self._dataset[5:6],
                                                                kind='wavnet',
                                                                fld=None,
                                                                blocksize=44100,
                                                                overlap=0,
                                                                n_mels=None,
                                                                power_melgram=None,
                                                                decibel_gram=None)

        self.assertEqual(data.shape, (5, 16*3 + 2))

class TestDFPreproc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._preprocessor = preprocess.Audio_Processor('../ESC-50/audio/')
        path_to_db='../ESC-50/'
        cls._dataset = pd.read_csv(path_to_db + 'meta/esc50.csv')
        classes = [None] * 50
        cls._dataset['h_category'] = None
        for index, row in cls._dataset.iterrows():
            target = row['target']
            classes[target] = row['category']
            cls._dataset.loc[index, 'h_category'] = mapping[row['category']]

    def test_dataframe_columns(self):
        data = self._preprocessor._Audio_Processor__preprocess_df(self._dataset[0:10],
                                                                kind='mfcc',
                                                                fld=None,
                                                                blocksize=220500,
                                                                overlap=0,
                                                                n_mels=128,
                                                                power_melgram=True,
                                                                decibel_gram=True)
        self.assertIn('l_target', data.columns)
        self.assertIn('h_target', data.columns)
        self.assertEqual(data.shape[0], 10)

if __name__ == '__main__':
    util_test = unittest.TestLoader().loadTestsFromTestCase(TestUtilities)
    unittest.TextTestRunner(verbosity=1).run(util_test)

    load_test = unittest.TestLoader().loadTestsFromTestCase(TestLoadAudio)
    unittest.TextTestRunner(verbosity=1).run(load_test)

    spec_test = unittest.TestLoader().loadTestsFromTestCase(TestSpecGet)
    unittest.TextTestRunner(verbosity=1).run(spec_test)

    mel_test = unittest.TestLoader().loadTestsFromTestCase(TestMelGet)
    unittest.TextTestRunner(verbosity=1).run(mel_test)

    mfcc_test = unittest.TestLoader().loadTestsFromTestCase(TestMFCCGet)
    unittest.TextTestRunner(verbosity=1).run(mfcc_test)

    wavnet_test = unittest.TestLoader().loadTestsFromTestCase(TestWavenetGet)
    unittest.TextTestRunner(verbosity=1).run(wavnet_test)

    preproc_test = unittest.TestLoader().loadTestsFromTestCase(TestDFPreproc)
    unittest.TextTestRunner(verbosity=1).run(preproc_test)
