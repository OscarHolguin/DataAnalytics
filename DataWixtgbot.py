import os
import telebot
from telebot import types
import asyncio

import pandas as pd
from datachat import generate_response_tg,generate_insights_one,generate_trends_and_patterns_one,aggregate_data,get_agent, get_insight_prompts,speech_to_text,create_pandas_dataframe_agent2
from io import BytesIO,BufferedReader
import nltk 
from tabulate import tabulate
import dataframe_image as dfi
BOT_TOKEN = "6314893362:AAGJWA1_fTiDU0q5I0XolGPmYBZyIMr1dBU"

bot = telebot.TeleBot(BOT_TOKEN)

last_file = {}
document_uploaded = False

#function to get pandas agent
def get_df(file,filename):
    file_ext = filename.split(".")[-1]
    if file_ext =='csv':
        df= pd.read_csv(BytesIO(file),encoding ="utf-8")
        return df
    elif file_ext =='xlsx':
        df= pd.read_excel(file)
        return df 

def get_suggestions(dfagent):
    suggestions = get_insight_prompts(dfagent)
    suggestions_s = [n for n in nltk.sent_tokenize(' '.join([x for x in nltk.word_tokenize(suggestions)]))]
    suggestions_s = [x for x in suggestions_s if x not in [str(n)+' .' for n in list(range(1,6))]]
    return suggestions_s




@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Hi, how are you doing? To proceed please send a document first")


@bot.message_handler(func=lambda msg: not document_uploaded)
def handle_non_document(message):
    bot.reply_to(message, "Please upload a document first.")


@bot.message_handler(func=lambda msg: document_uploaded)
def handle_other_messages(message):
    try:
        bot.reply_to(message, f"Getting information from your document.")
        data_response =  generate_response_tg(df,message.text,BOT_TOKEN,openail=True)
        if isinstance(data_response,str):
            #if the answer is text send the text message of it
            bot.reply_to(message,data_response)
        elif isinstance(data_response,BufferedReader):
            #if the answer is a graph send a photo of it
            with open("plot1.png", "rb") as plot:
                bot.send_photo(message.chat.id,plot)
    except Exception as e:
         bot.reply_to(message, f"Error {e}, please upload a document and try again.")



@bot.message_handler(content_types=['voice'])
def voice_processing(message):
    from pydub import AudioSegment
    bot.reply_to(message,"Transcribing audio please wait")

    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('useraudio.mp3', 'wb') as new_file:
        new_file.write(downloaded_file)
    
    transcribed_text = speech_to_text("useraudio.mp3")
    bot.reply_to(message,f"Running : {transcribed_text}")
    if not document_uploaded:
        bot.reply_to(message,"Upload a document first")
    else:
        data_response =  generate_response_tg(df,transcribed_text,BOT_TOKEN,openail=True)
        if isinstance(data_response,str):
            #if the answer is text send the text message of it
            bot.reply_to(message,data_response)
        elif isinstance(data_response,BufferedReader):
            #if the answer is a graph send a photo of it
            with open("plot1.png", "rb") as plot:
                bot.send_photo(message.chat.id,plot)


    
@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
        # Extract the callback data
    callback_data = call.data
    bot.send_message(call.message.chat.id,generate_response_tg(df,callback_data,BOT_TOKEN,openail=True))

@bot.message_handler(content_types=['document'])
def handle_document(message):
    global document_uploaded
    document_uploaded = True

    file_info = message.document
    file_name = file_info.file_name
    file_id = file_info.file_id
    last_file[message.chat.id] = file_id
    # Download the file
    file_path = bot.get_file(file_id).file_path
    downloaded_file = bot.download_file(file_path)
    #getting agent
    global df
    df = get_df(downloaded_file,file_name)
    global dfagent #Added for the suggestions menu part
    dfagent = get_agent(df)# this option is an agent without chat memory (stable version)
    #dfagent = create_pandas_dataframe_agent2(df)# this option is with memory
    bot.reply_to(message, f"Received file: {file_name} Ask me a question about this file")
    #suggestions1  =get_suggestions(dfagent)
    dftable = df.head().to_string()
    df_styled = df.head().style.background_gradient()
    bot.reply_to(message, f'<pre>{dftable}</pre>',parse_mode="HTML")
    dfi.export(df_styled,"mydf.png")
    with open("mydf.png",'rb') as dfpic:
        bot.send_photo(message.chat.id,dfpic)
    
    menuKeyboard = types.InlineKeyboardMarkup()
    suggestions1 = get_suggestions(dfagent)  # Replace with your actual suggestions
    for suggestion in suggestions1:
        menuKeyboard.add(types.InlineKeyboardButton(suggestion, callback_data=suggestion[:60]))
    bot.send_message(message.chat.id, "I can suggest the following: \n")    
    bot.send_message(message.chat.id, "Suggestions", reply_markup=menuKeyboard)


@bot.message_handler(commands=['options'])
def handle_suggestions(message):
    
    suggestions1 = get_suggestions(dfagent)  # Get your suggestions here
    # Create an inline keyboard
    menuKeyboard = types.InlineKeyboardMarkup()
    for suggestion in suggestions1:
        # Truncate long suggestions to fit the button text
        truncated_suggestion = suggestion[:60]
        menuKeyboard.add(types.InlineKeyboardButton(truncated_suggestion, callback_data=suggestion))

    # Send the suggestions to the user
    bot.send_message(message.chat.id, "I can suggest the following:", reply_markup=menuKeyboard)













#TB IMPLEMENTED
# @bot.message_handler(content_types=["audio"])
# def handle_audio(message):
#      bot.reply_to(message,"AUDIO FILE RECEIVED")
#      if not document_uploaded:
#          handle_non_document(message)
#      else:
#         #transcribe message
#         pass



    #MENU LIKE OPTIONS
    # bot.set_my_commands(commands=["menu"])


    # menuKeyboard = types.InlineKeyboardMarkup()
    # suggestions1  =get_suggestions(dfagent)
    # for suggestion in suggestions1:
    #     print(suggestion)
    #     menuKeyboard.add(types.InlineKeyboardButton(suggestion, callback_data=suggestion[:60]))
    


    
    #df.head().to_string(index=False))
    # for m in suggestions1:
    #     bot.reply_to(message,m)
    #call function that does everything

    #reply with 




if __name__ == "__main__":
    bot.polling()


