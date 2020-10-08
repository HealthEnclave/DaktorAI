import torch
import numpy as np
import face_detection
import subprocess
from os import path
import audio
import cv2
from gtts import gTTS
import os
import time
from pathlib import Path
from models.fatchord_version import WaveRNN
from models.forward_tacotron import ForwardTacotron
from utils import hparams as hp
from utils.text.symbols import phonemes
from utils.text import text_to_sequence, clean_text
from models_wav2lip import Wav2Lip
from tqdm import tqdm
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
    batch_size = 16
    while 1:
        predictions = []
        try:
            # for i in tqdm(range(0, len(images), batch_size)):
            for i in range(0, len(images), batch_size):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            logging.debug('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = [0, 10, 0, 0]
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results


def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection args.box was [-1, -1, -1, -1]

    for i, m in enumerate(mels):
        idx = i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (96, 96))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= 128:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, 96 // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, 96 // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    logging.debug("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


class VirtualDoctor:
    def __init__(self):
        self.temp = "temp"
        self.languages = ["english", "hindi"]
        hp.configure(os.path.join("pretrained", "pretrained_hparams.py"))

        self.voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                                 fc_dims=hp.voc_fc_dims,
                                 bits=hp.bits,
                                 pad=hp.voc_pad,
                                 upsample_factors=hp.voc_upsample_factors,
                                 feat_dims=hp.num_mels,
                                 compute_dims=hp.voc_compute_dims,
                                 res_out_dims=hp.voc_res_out_dims,
                                 res_blocks=hp.voc_res_blocks,
                                 hop_length=hp.hop_length,
                                 sample_rate=hp.sample_rate,
                                 mode=hp.voc_mode).to(device)

        voc_load_path = os.path.join("pretrained", "wave_575K.pyt")
        self.voc_model.load(voc_load_path)

        self.tts_model = ForwardTacotron(embed_dims=hp.forward_embed_dims,
                                         num_chars=len(phonemes),
                                         durpred_rnn_dims=hp.forward_durpred_rnn_dims,
                                         durpred_conv_dims=hp.forward_durpred_conv_dims,
                                         durpred_dropout=hp.forward_durpred_dropout,
                                         rnn_dim=hp.forward_rnn_dims,
                                         postnet_k=hp.forward_postnet_K,
                                         postnet_dims=hp.forward_postnet_dims,
                                         prenet_k=hp.forward_prenet_K,
                                         prenet_dims=hp.forward_prenet_dims,
                                         highways=hp.forward_num_highways,
                                         dropout=hp.forward_dropout,
                                         n_mels=hp.num_mels).to(device)

        tts_weights = os.path.join("pretrained", "forward_400K.pyt")
        self.tts_model.load(tts_weights)

    def text2hindi(self, text):
        lang_code = "hi"
        text_object = gTTS(text, lang=lang_code, slow=False)
        audio_path = Path(os.path.join(self.temp, "audio.wav"))
        text_object.save(audio_path)
        return audio_path

    def text2english(self, text):
        text = clean_text(text.strip())
        inputs = [text_to_sequence(text)]
        audio_path = Path(os.path.join(self.temp, "audio.wav"))
        for i, x in enumerate(inputs, 1):
            logging.debug(f'\n| Generating {i}/{len(inputs)}')
            _, m, _ = self.tts_model.generate(x, alpha=1)

            m = torch.tensor(m).unsqueeze(0)
            self.voc_model.generate(m, audio_path, hp.voc_gen_batched, hp.voc_target, hp.voc_overlap, hp.mu_law)
        return audio_path

    def speech2lip(self, face, _audio, outfile):
        model = load_model(os.path.join("face_detection", "detection", "sfd", "wav2lip_gan.pth"))
        resize_factor = 4  # 480p or 720p best
        video_stream = cv2.VideoCapture(face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        logging.debug('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))

            y1, y2, x1, x2 = [0, -1, 0, -1]
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

        logging.debug("Number of frames available for inference: " + str(len(full_frames)))

        wav = audio.load_wav(path.relpath(_audio), 16000)
        mel = audio.melspectrogram(wav)
        logging.debug(mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + 16 > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - 16:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + 16])
            i += 1

        logging.debug("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[:len(mel_chunks)]

        gen = datagen(full_frames.copy(), mel_chunks)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter('temp/result.avi',
                                      cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()

        command = 'ffmpeg -loglevel panic -y -i {} -i {} -strict -2 -q:v 1 {}'.format(_audio, 'temp/result.avi', outfile)
        subprocess.call(command, shell=True)

    def main(self, txt_file, language):
        if not isinstance(txt_file, str):
            raise TypeError(f"txt_file must be type str, got: {type(txt_file)} ({txt_file})")
        if not isinstance(language, str):
            raise TypeError(f"{language} must be type str, got: {type(language)} ({language})")
        if language not in self.languages:
            raise ValueError(f"Language {language} is not supported! Language {language} must be in {self.languages}.")

        text = open(txt_file).read()
        if language == "english":
            audio_path = self.text2english(text)
        else:
            audio_path = self.text2hindi(text)

        video_path = "./examples/doctor.mp4"
        result_path = "result.mp4"

        self.speech2lip(video_path, audio_path, result_path)

        return result_path


if __name__ == '__main__':
    logger.info("Starting script....")
    logger.info("Initialising Virtual Doctor....")
    VD = VirtualDoctor()
    logger.info("Successfully initialised Virtual Doctor....")
    text_file = "test.txt"
    language = "english"
    logger.info(f"Making video using text from {text_file} in {language} language....")
    start_time = time.time()
    result = VD.main(text_file, language)
    logger.info(f"Finished making the Virtual Doctor video. Time needed to produce it: {time.time() - start_time} seconds. You can find it under {result}")