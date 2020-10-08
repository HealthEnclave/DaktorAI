import torch
from models.fatchord_version import WaveRNN
from models.forward_tacotron import ForwardTacotron
from utils import hparams as hp
from utils.text.symbols import phonemes
from utils.text import text_to_sequence, clean_text


def voice(text):
    hp.configure("pretrained/pretrained_hparams.py")

    tts_weights = "pretrained/forward_400K.pyt"

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Using device:', device)
    print('\nInitialising WaveRNN Model...\n')
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
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

    voc_load_path = "pretrained/wave_575K.pyt"
    voc_model.load(voc_load_path)

    print('\nInitialising Forward TTS Model...\n')
    tts_model = ForwardTacotron(embed_dims=hp.forward_embed_dims,
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

    tts_model.load(tts_weights)

    text = clean_text(text.strip())
    inputs = [text_to_sequence(text)]

    for i, x in enumerate(inputs, 1):
        print(f'\n| Generating {i}/{len(inputs)}')
        _, m, _ = tts_model.generate(x, alpha=1)

        save_path = "output.wav"
        m = torch.tensor(m).unsqueeze(0)
        voc_model.generate(m, save_path, hp.voc_gen_batched, hp.voc_target, hp.voc_overlap, hp.mu_law)


input_text = "Hey this is really interesting to run. I dont know if it performs good. How are you today? Im really good and want to be loved!"
voice(input_text)
