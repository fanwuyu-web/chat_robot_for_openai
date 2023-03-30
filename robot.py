# coding=utf-8
import baiduApi
import gptApi
from playsound import playsound
import gptApi_duolun
from VITS.VC_inference import getVoice
from dotenv.main import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()

    api_id = os.environ['api_id']
    api_key = os.environ['api_key']
    api_secert = os.environ['api_secert']
    bdr = baiduApi.BaiduRest(api_id, api_key, api_secert)
    # robot = gptApi.GptApi()
    messages = [{"role": "system", "content": "你现在是很有用的助手！"}]
    while True:
        input("按下回车开始说话，自动停止")
        print('开始录音')
        bdr.recorder("./input/input.wav")
        print("结束")
        ask = bdr.getText('./input/input.wav')
        print('你：', ask)

        # robot=gptApi.GptApi()
        # ans = robot.test_openai(ask)
        messages.append({"role": "user", "content": ask})
        ans = gptApi_duolun.generate_answer(messages)
        messages.append({"role": "assistant", "content": ans})
        print('机器人：', ans)

        #bdr.getVoice(ans, "./output.mp3")#调用百度api合成语音，已经弃用
        getVoice(ans.lstrip(),'./output/output.wav')#默认路径：./output/output.mp3,同时删除文本前面空格，以免无法生成语音
        #bdr.speakMac("./output.mp3")
        playsound('./output/output.wav')
