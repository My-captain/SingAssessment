# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-02-19 18:
@file: preprocess.py
"""
import collections
import contextlib
import csv
import json
import math
import os
import sys
import wave

import crepe
import librosa
import soundfile
import tqdm
import webrtcvad
import numpy as np


def generate_meta_json():
    meta_data = []
    reader = csv.reader(open("副本-集美项目评分一期 -导入信息.csv", "r"))
    next(reader)
    for line in reader:
        meta_data.append({
            "offset": line[1],
            "fileId": line[2].split(".")[0].split("/")[1],
            "referenceSong": line[3].split("/")[1],
            "referenceText": line[4].split("/")[1],
            "pitch": float(line[6]),
            "rhythm": float(line[7]),
            "pronunciation": float(line[8]),
        })
        print(line)
    json.dump(meta_data, open("meta_data.json", "w"), ensure_ascii=False)


def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """
    按序检测出帧列表中的声音事件，并组合返回
    :param sample_rate: 采样率
    :param frame_duration_ms: 帧长
    :param padding_duration_ms: 帧padding
    :param vad: VAD对象
    :param frames: 帧列表
    :return:
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        sys.stdout.write('1' if vad.is_speech(frame.bytes, sample_rate) else '0')
        if not triggered:
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer if vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('+(%s)' % (ring_buffer[0].timestamp,))
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            num_unvoiced = len([f for f in ring_buffer if not vad.is_speech(f.bytes, sample_rate)])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def split_clips(chunk_path, clip_duration=3, min_limit=3, tail_sample_duration=0.3):
    # song = AudioSegment.from_wav(chunk_path)
    # if len(song) < min_limit*1000:
    #     return None
    # else:
    #     clips = []
    #     whole_clip_count = len(song) // (clip_duration*1000)
    #     for i in range(whole_clip_count):
    #         clips.append(song[i*clip_duration*1000:(i+1)*clip_duration*1000])
    #         # print(i*clip_duration*1000, (i+1)*clip_duration*1000)
    #     if len(song) - whole_clip_count*clip_duration*1000 > tail_sample_duration*1000:
    #         # print(-clip_duration*1000+1)
    #         clips.append(song[-clip_duration*1000:])
    #     return clips
    y, sr = librosa.load(chunk_path, sr=16000)
    duration = librosa.get_duration(y, sr)
    if duration < min_limit:
        return None
    else:
        clips = []
        whole_clip_count = math.floor(duration / clip_duration)
        for i in range(whole_clip_count):
            clips.append(y[i * clip_duration * sr:(i + 1) * clip_duration * sr])
            # print(i*clip_duration*1000, (i+1)*clip_duration*1000)
        if duration - whole_clip_count * clip_duration > tail_sample_duration:
            # print(-clip_duration*1000+1)
            clips.append(y[-clip_duration * sr:])
        return clips


def save_clip_dataset(vad_path, song_scores, out_path):
    clip_meta = []
    for item in tqdm.tqdm(os.listdir(vad_path)):
        item_path = os.path.join(vad_path, item)
        if os.path.isfile(item_path):
            continue
        if item not in song_scores.keys():
            continue
        song_id = item
        for chunk in os.listdir(item_path):
            chunk_path = os.path.join(item_path, chunk)
            chunk_name = chunk.split(".")[0]
            clips = split_clips(chunk_path)
            if clips is None:
                continue
            else:
                for clip_seq, clip in enumerate(clips):
                    clip_id = f"song-{song_id}_{chunk_name}_clip-{clip_seq}.wav"
                    song_score = song_scores[item]
                    clip_meta.append({
                        "clipId": clip_id,
                        "songId": song_id,
                        "songScore": song_score
                    })
                    soundfile.write(os.path.join(out_path, clip_id), clip, 16000)
    return clip_meta


def generate_clip_dataset():
    quality_400_meta = json.load(open("/home/zliu-elliot/workspace/SingAssessment/data/quality_400/meta_data.json", "r"))
    song_scores = dict()
    for song in quality_400_meta:
        audio_id = song["student_audio"].split("/")[1].split(".")[0]
        song_scores[audio_id] = song["scores"][-1]
    meta = save_clip_dataset("/home/zliu-elliot/workspace/SingAssessment/data/quality_400/vad", song_scores, "/home/zliu-elliot/workspace/SingAssessment/data/quality_400/clips")
    json.dump(meta, open("quality_400/clips_metadata.json", "w"), ensure_ascii=False)


def split_metadata():
    metajson = json.load(open("quality_400/clips_metadata.json", "r"))
    idx = [i for i in range(len(metajson))]
    np.random.shuffle(idx)

    train = math.floor(len(idx)*0.8)
    test = math.floor(len(idx)*0.9)
    train = idx[:train]
    valid = idx[train:test]
    test = idx[-test:]

    train_meta = [metajson[i] for i in train]
    valid_meta = [metajson[i] for i in valid]
    test_meta = [metajson[i] for i in test]

    json.dump(train_meta, open("quality_400/clips_metadata_train.json", "w"), ensure_ascii=False)
    json.dump(valid_meta, open("quality_400/clips_metadata_valid.json", "w"), ensure_ascii=False)
    json.dump(test_meta, open("quality_400/clips_metadata_test.json", "w"), ensure_ascii=False)


def seq_2_matrix(seq, dimension):
    seq_norm = (seq - np.min(seq)) / (np.max(seq) - np.min(seq))
    matrix = np.zeros((dimension, seq.shape[0]))
    for idx, freq in enumerate(seq_norm):
        matrix[math.floor(freq*96)-1, idx] = 1
    return matrix


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y, sr, hop_length=512, n_mels=96)
    log_mel = librosa.power_to_db(mel_spectrogram)
    tempogram = librosa.feature.tempogram(y, sr, win_length=96)
    chroma = librosa.feature.chroma_stft(y, sr, n_chroma=96)
    # time, frequency, confidence, f0_activation = crepe.predict(y, sr, viterbi=True, step_size=32, center=False)
    f0_seq = librosa.yin(y, fmin=80, fmax=400, sr=sr)
    f0_matrix = seq_2_matrix(f0_seq, 96)
    return np.array((log_mel, chroma, f0_matrix, tempogram))


def save_feature_dataset(clip_dir="./quality_400/clips", out_dir="./quality_400/clip_feature"):
    for item in tqdm.tqdm(os.listdir(clip_dir)):
        item_name = item.split(".")[0]
        item_path = os.path.join(clip_dir, item)
        features = extract_features(item_path)
        np.save(os.path.join(out_dir, f"{item_name}.npy"), features)


def main(dataset_path):
    for i in os.listdir(dataset_path):
        item_path = f"{dataset_path}{i}"
        y, sr = librosa.load(f"{item_path}/vocals.wav")
        y = librosa.resample(y, sr, 16000)
        soundfile.write(f"{item_path}/resampled.wav", y, 16000)
        audio, sample_rate = read_wave(f"{item_path}/resampled.wav")
        os.mkdir(f"{item_path}/vad")
        vad = webrtcvad.Vad(0)
        frames = frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, 30, 300, vad, frames)
        for i, segment in enumerate(segments):
            path = f"{item_path}/vad/chunk-{i}.wav"
            print(' Writing %s' % (path,))
            write_wave(path, segment, sample_rate)


if __name__ == '__main__':
    # generate_meta_json()
    # main("/home/zliu-elliot/workspace/SingAssessment/audio_output/")
    # generate_clip_dataset()
    # save_feature_dataset()
    split_metadata()
