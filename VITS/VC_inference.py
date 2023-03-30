import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
from VITS import commons#自建文件
#from mel_processing import spectrogram_torch#自建文件
from VITS import utils#自建文件
from VITS.models import SynthesizerTrn#自建文件
import librosa
from playsound import playsound
import logging


from VITS.text import text_to_sequence, _clean_text

from scipy.io.wavfile import write

def to_wav(audio,path):
    #将ndarray转换为int16数据类型（必须为16位有符号整数）
    # data = audio * 32767
    # data = data.astype(np.int16)
    data=audio
    # 设置采样率
    sample_rate = 24100

    # 将数据写入.wav文件
    write(path, sample_rate, data)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}
lang = ['日本語', '简体中文', 'English', 'Mix']
def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, language, speed,path):
        if language is not None:
            text = language_marks[language] + text + language_marks[language]#选择推理语言
        speaker_id = speaker_ids[speaker]#选择说话的人的id
        stn_tst = get_text(text, hps, False)#获取文本并选择模型文件（hps）
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            #src="http://127.0.0.1:7860/file=C:\Users\15093289086\AppData\Local\Temp\tmpjz4wxooq.wav"
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()#
            ##print(type(audio))
            to_wav(audio,path)

        del stn_tst, x_tst, x_tst_lengths, sid
        ##print("Success", (hps.data.sampling_rate, audio))
        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn

# def create_vc_fn(model, hps, speaker_ids):
#     def vc_fn(original_speaker, target_speaker, record_audio, upload_audio):
#         input_audio = record_audio if record_audio is not None else upload_audio
#         if input_audio is None:
#             return "You need to record or upload an audio", None
#         sampling_rate, audio = input_audio
#         original_speaker_id = speaker_ids[original_speaker]
#         target_speaker_id = speaker_ids[target_speaker]
#
#         audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
#         if len(audio.shape) > 1:
#             audio = librosa.to_mono(audio.transpose(1, 0))
#         if sampling_rate != hps.data.sampling_rate:
#             audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
#         with no_grad():
#             y = torch.FloatTensor(audio)
#             y = y / max(-y.min(), y.max()) / 0.99
#             y = y.to(device)
#             y = y.unsqueeze(0)
#             spec = spectrogram_torch(y, hps.data.filter_length,
#                                      hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
#                                      center=False).to(device)
#             spec_lengths = LongTensor([spec.size(-1)]).to(device)
#             sid_src = LongTensor([original_speaker_id]).to(device)
#             sid_tgt = LongTensor([target_speaker_id]).to(device)
#             audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
#                 0, 0].data.cpu().float().numpy()
#         del y, spec, spec_lengths, sid_src, sid_tgt
#         return "Success", (hps.data.sampling_rate, audio)
#
#     return vc_fn



def getVoice(text,path):
    logging.getLogger('jieba').disabled = True#禁用日志输出
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="VITS/G_latest.pth", help="directory to your fine-tuned model")
    parser.add_argument("--config_dir", default="VITS/finetune_speaker.json", help="directory to your model config file")
    parser.add_argument("--share", default=False, help="make link public (used in colab)")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config_dir)

    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model_dir, net_g, None)
    speaker_ids = hps.speakers
    speakers = list(hps.speakers.keys())
    tts_fn = create_tts_fn(net_g, hps, speaker_ids)
    # vc_fn = create_vc_fn(net_g, hps, speaker_ids)

    # 本地运行，不挂web
    textbox = text
    char_dropdown = speakers[0]
    language_dropdown = lang[1]
    duration_slider = 0.8

    tts_fn(textbox, char_dropdown, language_dropdown, duration_slider,path)

    #print('okk...')
    # playsound('../output/output.wav')

if __name__ == "__main__":

    #logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger('jieba').disabled = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./G_latest.pth", help="directory to your fine-tuned model")
    parser.add_argument("--config_dir", default="./finetune_speaker.json", help="directory to your model config file")
    parser.add_argument("--share", default=False, help="make link public (used in colab)")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config_dir)


    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model_dir, net_g, None)
    speaker_ids = hps.speakers
    speakers = list(hps.speakers.keys())
    tts_fn = create_tts_fn(net_g, hps, speaker_ids)
    #vc_fn = create_vc_fn(net_g, hps, speaker_ids)

    #本地运行，不挂web
    textbox = "你好，博士。"
    char_dropdown = speakers[0]
    language_dropdown = lang[1]
    duration_slider = 0.8

    tts_fn(textbox, char_dropdown, language_dropdown, duration_slider, './output.wav')

    print('okk...')
    playsound('../output/output.wav')


    # app = gr.Blocks()
    # with app:
    #     with gr.Tab("Text-to-Speech"):
    #         with gr.Row():
    #             with gr.Column():
    #                 textbox = gr.TextArea(label="Text",
    #                                       placeholder="Type your sentence here",
    #                                       value="博士，很高兴见到你，欢迎回来。", elem_id=f"tts-input")
    #                 # select character
    #                 char_dropdown = gr.Dropdown(choices=speakers, value=speakers[0], label='character')
    #                 #print(char_dropdown)
    #                 language_dropdown = gr.Dropdown(choices=lang, value=lang[1], label='language')
    #                 #print(language_dropdown)
    #                 duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1,
    #                                             label='速度 Speed')
    #             with gr.Column():
    #                 text_output = gr.Textbox(label="Message")
    #                 audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
    #                 btn = gr.Button("Generate!")
    #                 #res=tts_fn(textbox, char_dropdown, language_dropdown, duration_slider)
    #                 ##print(res)
    #                 btn.click(tts_fn,#发包函数
    #                           inputs=[textbox, char_dropdown, language_dropdown, duration_slider,],
    #                           outputs=[text_output, audio_output])#这是绑定了函数
    #
    #             #可以写成tts_fn(textbox, char_dropdown, language_dropdown, duration_slider)
    #
    #     with gr.Tab("Voice Conversion"):#这个是声音转声音的
    #         gr.Markdown("""
    #                         录制或上传声音，并选择要转换的音色。User代表的音色是你自己。
    #         """)
    #         with gr.Column():
    #             record_audio = gr.Audio(label="record your voice", source="microphone")
    #             upload_audio = gr.Audio(label="or upload audio here", source="upload")
    #             source_speaker = gr.Dropdown(choices=speakers, value="User", label="source speaker")
    #             target_speaker = gr.Dropdown(choices=speakers, value=speakers[0], label="target speaker")
    #         with gr.Column():
    #             message_box = gr.Textbox(label="Message")
    #             converted_audio = gr.Audio(label='converted audio')
    #         btn = gr.Button("Convert!")
    #         btn.click(vc_fn, inputs=[source_speaker, target_speaker, record_audio, upload_audio],
    #                   outputs=[message_box, converted_audio])
    # webbrowser.open("http://127.0.0.1:7860")
    # app.launch(share=args.share)
    #
