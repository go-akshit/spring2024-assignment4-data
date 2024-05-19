from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

from fastwarc.warc import ArchiveIterator, WarcRecordType
import fasttext
import regex as re
import random
import gzip
import math
from nltk import word_tokenize

def extract_text(html_byte_str : bytes):
    encoding = detect_encoding(html_byte_str)
    decoded_string = html_byte_str.decode(encoding)
    return extract_plain_text(decoded_string)

def extract_text_from_warc(warc_file_path : str, output_file_path : str):
    i = 0
    
    for record in ArchiveIterator(open(warc_file_path, 'rb')):
        if (record.record_type == WarcRecordType.response) and ('text/html' in record.http_headers.get('Content-Type')):
            record_content = record.reader.read()
            
            text = extract_text(record_content)
            
            i += 1
        if i == 1:
            break
    
    with open(output_file_path, 'w') as f:
        f.write(text)


def language_detection(text : str):
    model = fasttext.load_model('/home/shared/lid.176.bin')
    text = text.replace('\n', ' ')
    prediction =  model.predict(text)
    language = prediction[0][0].replace('__label__', '')
    confidence = prediction[1][0]
    return language, confidence


def mask_email(text : str):
    email_pattern = r'(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"\\(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])'
    email_addresses = re.findall(email_pattern, text)
    masked_text = re.sub(email_pattern, '|||EMAIL_ADDRESS|||', text)
    return masked_text, len(email_addresses)

def mask_phone_numbers(text : str):
    phone_number_pattern = re.compile(r'''                                    
    # Area code (optional, with parentheses or without)
    (\(?\d{3}\)|\d{3})
    # Optional separator (space, dot, or dash)
    [\s?.-]?
    # Next 3 digits
    \d{3}
    # Optional separator (space, dot, or dash)
    [\s?.-]?
    # Last 4 digits
    \d{4}
    ''', re.VERBOSE)
    phone_numbers = re.findall(phone_number_pattern, text)
    masked_text = re.sub(phone_number_pattern, '|||PHONE_NUMBER|||', text)
    return masked_text, len(phone_numbers)

def mask_ips(text : str):
    ip_address_pattern = re.compile(r'''
    # First 3 octets
    (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.
    (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.
    (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.
    # Last octet
    (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)
    ''', re.VERBOSE)
    ip_addresses = re.findall(ip_address_pattern, text)
    masked_text = re.sub(ip_address_pattern, '|||IP_ADDRESS|||', text)
    return masked_text, len(ip_addresses)

def classify_nsfw(text : str):
    nsfw_model = fasttext.load_model('/home/shared/dolma-jigsaw-fasttext-bigrams-nsfw.bin')
    model_prediction = nsfw_model.predict(text)
    label = model_prediction[0][0].replace('__label__', '')
    confidence = model_prediction[1][0]
    return label, confidence

def classify_toxic_speech(text : str):
    toxic_model = fasttext.load_model('/home/shared/dolma-jigsaw-fasttext-bigrams-hatespeech.bin')
    model_prediction = toxic_model.predict(text)
    label = model_prediction[0][0].replace('__label__', '')
    confidence = model_prediction[1][0]
    return label, confidence

def gopher_quality_filter(text : str):
    tokens = word_tokenize(text)

    min_num_tokens = 50
    max_num_tokens = 100_000

    min_avg_word_len = 3
    max_avg_word_len = 10

    #token count check
    num_tokens = len(tokens)
    if num_tokens < min_num_tokens or len(tokens) > max_num_tokens:
        return False
    
    word_len = 0
    alphabet_words = 0
    for word in tokens:
        word_len += len(word)
        for char in word:
            if char.isalpha():
                alphabet_words += 1
                break
    
    #average word length check
    avg_word_len = word_len / num_tokens
    if avg_word_len < min_avg_word_len or avg_word_len > max_avg_word_len:
        return False
    
    #alphabet words check
    if alphabet_words/num_tokens < 0.8:
        return False
    

    #ellipsis check
    lines = text.split('\n')
    num_lines = len(lines)
    ellipsis_count = 0
    for line in lines: 
        if line.strip().endswith('...'): 
            ellipsis_count += 1
    if ellipsis_count/num_lines > 0.3:
        return False
    
    return True


def reservoir_sampling(input_file_path : str, sample_size : int, output_file_path : str):
    sample = []
    if('.gz' in input_file_path):
        with gzip.open(input_file_path, 'rt', encoding='utf-8') as f:
            for i in range(sample_size):
                line = f.readline().strip()
                sample.append(line)
            
            
            w = math.exp(math.log(random.random()) / sample_size)
            for i, line in enumerate(f, start=sample_size+1):
                skip = math.floor(math.log(random.random()) / math.log(1 - w))
                i += skip + 1
                print(i)
                for _ in range(skip):
                    f.readline()
                
                line = f.readline().strip()

                if line:
                    sample[random.randint(0, sample_size-1)] = line
                w *= math.exp(math.log(random.random()) / sample_size)
    
    with open(output_file_path, 'w') as f:
        for line in sample:
            f.write(line)
            f.write('\n')

def main():
    
    # extract_text_from_warc('../CC-MAIN-20180420081400-20180420101400-00118.warc.gz', 'output_extract_text.txt')
    # with open('output_extract_text.txt', 'r') as f:
    #     merged_line = ""
    #     for i in range(30):
    #         line = f.readline()
    #         line = line.replace('\xa0', ' ')
    #         line = line.strip("\n ")
    #         merged_line = merged_line + line
    #     import pdb; pdb.set_trace()
    #     language, confidence = language_detection(merged_line)
    #     print(language, confidence)

    

    # masked_text, email_count = mask_phone_numbers(input_str)
    # print(masked_text)
    # print(email_count)


    # label, confidence = classify_nsfw('this is a test. fing someone')
    # print(label, confidence)

    # label, confidence = classify_toxic_speech('this is a test.')
    # print(label, confidence)

    #gopher_quality_filter('this is a test.')

    reservoir_sampling('/home/shared/enwiki-20240420-extracted_urls.txt.gz', 2, 'positive_url_sample.txt')
if __name__ == '__main__':
    main()