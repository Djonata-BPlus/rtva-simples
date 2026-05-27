import json
import tempfile
import wave
import pyaudio
from groq import Groq
import asyncio
import edge_tts
import pygame


client = Groq(
    api_key= "Busque no .env.production"
)

MODEL_STT = "whisper-large-v3" #modelo da groq para fazer o STT
MODEL_LLM = "openai/gpt-oss-20b" #llm para "pensar" na resposta
VOICE = "pt-BR-FranciscaNeural"

##Configs padrão
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5


def verificar_status_os(os_numero):
    print(f"\n[SISTEMA]: Consultando status da OS {os_numero}")

    ordens = {
        "1": "ABERTO",
        "2": "FECHADO",
        "3": "EM ANDAMENTO"
    }

    status = ordens.get(str(os_numero))

    if status:
        return f"O status da OS {os_numero} é {status}."

    return f"OS {os_numero} não encontrada."


def gravar_audio():
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("\nOuvindo...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Processando...")

    stream.stop_stream()
    stream.close()
    p.terminate()

    temp_wav = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    with wave.open(temp_wav.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return temp_wav.name

#faz o STT
def transcrever_audio(audio_path):
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model=MODEL_STT,
            language="pt"
        )

    return transcription.text


#filtro de ruído STT, adicionado com base nos erros do stt atual usado que alucinava justamente com essas palavras...
def texto_invalido(texto):
    texto_limpo = texto.strip().lower()

    if len(texto_limpo) < 3:
        return True

    frases_ruins = [
        "legenda",
        "obrigado",
        "valeu",
        "thank you",
        "thanks for watching",
        "music",
        "subscribe"
    ]

    return any(frase in texto_limpo for frase in frases_ruins)

async def falar_async(texto):
    output_file = "voz.mp3"

    communicate = edge_tts.Communicate(
        texto,
        VOICE,
        rate="-5%"
    )

    await communicate.save(output_file)

    pygame.mixer.init()
    pygame.mixer.music.load(output_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)

    pygame.mixer.quit()

def falar(texto):
    asyncio.run(falar_async(texto))

def pipeline_conversa():
    historico = [
        {
            "role": "system",
            "content": """
Você é um agente de voz para consulta de ordens de serviço.

REGRAS:
- Responda sempre em português.
- Responda curto.
- Máximo 1 frase.
- Seja natural.
- Nunca escreva XML.
- Nunca escreva tags <function>.
- Nunca invente chamadas de função.
- Use verificar_status_os apenas quando o usuário perguntar sobre OS,
ordem de serviço ou status.
- Só use verificar_status_os quando o usuário informar claramente o
número da OS, e envie por parâmetro apenas o número, sempre uma string
com um número, nunca um objeto, exemplo "1" e não "OS1", .
- Se o usuário disser tchau, obrigado, olá ou frases genéricas,
responda normalmente sem usar ferramenta.
- Se não entender o áudio diga: "Não entendi bem, pode repetir?"
"""
        }
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "verificar_status_os",
                "description": "Consulta o status de uma ordem de
serviço quando o usuário informa o número da OS",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "os": {
                            "type": "string",
                            "description": "Número da ordem de
serviço. Exemplo: 1, 2, 3"
                        }
                    },
                    "required": ["os"]
                }
            }
        }
    ]

    while True:
        try:
            input("\nPressione ENTER para gravar sua fala...")

            audio_path = gravar_audio()
            texto_usuario = transcrever_audio(audio_path)

            print(f"\nTranscrição: {texto_usuario}")

            if texto_invalido(texto_usuario):
                print("Ruído/silêncio identificado e ignorado.")
                continue

            historico.append({
                "role": "user",
                "content": texto_usuario
            })

            response = client.chat.completions.create(
                model=MODEL_LLM,
                messages=historico,
                tools=tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            if response_message.tool_calls:
                historico.append({
                    "role": "assistant",
                    "tool_calls": response_message.tool_calls
                })

                for tool_call in response_message.tool_calls:
                    print(f"\n o que retornou {tool_call.function.arguments}")
                    args = json.loads(tool_call.function.arguments)

                    os_numero = args.get("os")

                    if not os_numero:
                        print("\nAgente: Não encontrei o número da OS.
Pode repetir?")
                        continue

                    resultado = verificar_status_os(os_numero)

                    historico.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": resultado
                    })

                segunda_res = client.chat.completions.create(
                    model=MODEL_LLM,
                    messages=historico
                )

                final_text = segunda_res.choices[0].message.content

                print(f"\nAgente: {final_text}")
                falar(final_text)

                historico.append({
                    "role": "assistant",
                    "content": final_text
                })

            else:
                print(f"\nResposta para fala: {response_message.content}")
                falar(response_message.content)
                historico.append({
                    "role": "assistant",
                    "content": response_message.content
                })

        except KeyboardInterrupt:
            print("\nEncerrando...")
            break


pipeline_conversa()
