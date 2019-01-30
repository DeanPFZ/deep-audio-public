import unittest
import context
import json
import preprocess
import pandas as pd
from unittest import mock
from random import randint

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
                                                                fld=1, blocksize=55125, overlap=0)
        self.assertEqual(samples.shape, (35, 1, 55125))
        self.assertEqual(len(h_category), 35)
        self.assertEqual(len(l_category), 35)

    def test_load_fld_audio_w_blocksize_w_overlap(self):
        samples, h_category, l_category = self._preprocessor._Audio_Processor__load_audio(
                                                                data=self._dataset[:10],
                                                                fld=1, blocksize=55125, overlap=4096)
        self.assertEqual(samples.shape, (42, 1, 55125))
        self.assertEqual(len(h_category), 42)
        self.assertEqual(len(l_category), 42)

    def test_load_audio_wo_blocksize(self):
        random = randint(0,50)
        samples, h_category, l_category = self._preprocessor._Audio_Processor__load_audio(
                                                                data=self._dataset[random:random+10],
                                                                fld=None, blocksize=0, overlap=0)
        self.assertEqual(samples.shape, (10, 220500))
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

# TODO: Create unit tests for spectrogram get
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

# TODO: Create unit tests for melgram get
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

# TODO: Create unit tests for mfcc encoding
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

# TODO: Create unit tests for wavenet encoding
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

if __name__ == '__main__':
    util_test = unittest.TestLoader().loadTestsFromTestCase(TestUtilities)
    unittest.TextTestRunner(verbosity=2).run(util_test)

    load_test = unittest.TestLoader().loadTestsFromTestCase(TestLoadAudio)
    unittest.TextTestRunner(verbosity=2).run(load_test)
