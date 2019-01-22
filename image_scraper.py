import json 
import requests 
from bs4 import BeautifulSoup
import html2text
import datetime
import rfc3339
import os
import uuid
from googleapiclient.discovery import build
from googleapiclient.errors import Error
import time
import io
from google.cloud import vision
import pandas as pd 
# from pdf2image import convert_from_path

credential_path = "TUCKER-krow-network-1533419444055-32d5a289781e.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
os.environ['GOOGLE_CLOUD_PROJECT'] = "krow-network-1533419444055"

parent = 'projects/' + os.environ['GOOGLE_CLOUD_PROJECT']
# client_service = build('vision', 'v1')



def detect_document(path):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

 
    # print (content)
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
        # print (content)

    image = vision.types.Image(content=content)

    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        # print (len(page.blocks))
        # # exit()
        # for block in page.blocks:
        #     # print('\nBlock confidence: {}\n'.format(block.confidence))
        #     p = ""
        #     for paragraph in block.paragraphs:
        #         # print('Paragraph confidence: {}'.format(
        #         #     paragraph.confidence))
        #         print (block.bounding_box)
        #         p = ""
        #         for word in paragraph.words:
        #             word_text = ''.join([
        #                 symbol.text for symbol in word.symbols
        #             ])
        #             p += word_text + " "
        #             # print('Word text: {} (confidence: {})'.format(
        #             #     word_text, word.confidence))
        #     print (p + "\n\n")

        #         #     for symbol in word.symbols:
        #         #         print('\tSymbol: {} (confidence: {})'.format(
        #         #             symbol.text, symbol.confidence))

        breaks = vision.enums.TextAnnotation.DetectedBreak.BreakType

        paragraphs = []
        lines = []
        for block in page.blocks:
            for paragraph in block.paragraphs:
                para = ""
                line = ""
                for word in paragraph.words:
                    for symbol in word.symbols:
                        line += symbol.text
                        if symbol.property.detected_break.type == breaks.SPACE:
                            line += ' '
                        if symbol.property.detected_break.type == breaks.EOL_SURE_SPACE:
                            line += ' '
                            lines.append(line)
                            para += line
                            line = ''
                        if symbol.property.detected_break.type == breaks.LINE_BREAK:
                            lines.append(line)
                            para += line
                            line = ''
                paragraphs.append(para)
        return " ".join(paragraphs)
        # print(lines)


