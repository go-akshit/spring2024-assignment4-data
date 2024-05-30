from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

from fastwarc.warc import ArchiveIterator, WarcRecordType
import fasttext
import regex as re
import random
import gzip
import math
from nltk import word_tokenize
from pathlib import Path
import mmh3
import string
import unicodedata
import re
import os
import submitit
from tqdm import tqdm

def extract_text(html_byte_str : bytes):
    encoding = detect_encoding(html_byte_str)
    decoded_string = html_byte_str.decode(encoding, 'ignore')
    return extract_plain_text(decoded_string)

def extract_text_from_warc(warc_file_path : str, output_file_path : str):
    i = 0
    j = 0
    with open(output_file_path, 'w') as f:
        for record in ArchiveIterator(open(warc_file_path, 'rb')):
            if (record.record_type == WarcRecordType.response):
                if (record.http_headers.get('Content-Type') and 'text/html' in record.http_headers.get('Content-Type')):
                    record_content = record.reader.read()
                    text = extract_text(record_content)
                    text, _ = mask_email(text)
                    text, _ = mask_phone_numbers(text)
                    text, _ = mask_ips(text)
                    i += 1
                    print('i = ', i)
                    if(gopher_quality_filter(text) == False):
                        continue
                    text = text.replace('\n', ' ')
                    if(language_detection(text)[0] != 'en'):
                        continue
                    text = re.sub(r'[^\x00-\x7F]+', '', text)
                    text = normalize_text(text)
                    # if(classify_nsfw(text)[0] != 'non-nsfw'):
                    #     continue
                    # if(classify_toxic_speech(text)[0] != 'non-toxic'):
                    #     continue                
                    j += 1
                    print('j = ', j)
                    f.write('__label__bad')
                    f.write(' ')
                    f.write(text)
                    f.write('\n')
            if j == 12000:
                break
    
    
    


def language_detection(text : str):
    model = fasttext.load_model('../../lid.176.bin')
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
    nsfw_model = fasttext.load_model('../../nsfw_model.bin')
    model_prediction = nsfw_model.predict(text)
    label = model_prediction[0][0].replace('__label__', '')
    confidence = model_prediction[1][0]
    return label, confidence

def classify_toxic_speech(text : str):
    toxic_model = fasttext.load_model('../../hatespeech_model.bin')
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

def reservoir_sampling(input_file_path : str, sample_size : int, output_file_path1 : str, output_file_path2):
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
    
    sample_first_half = sample[:sample_size//2]
    sample_second_half = sample[sample_size//2:]
    
    with open(output_file_path1, 'w') as f:
        for line in sample_first_half:
            f.write(line)
            f.write('\n')

    with open(output_file_path2, 'w') as f:
        for line in sample_second_half:
            f.write(line)
            f.write('\n')

def exact_deduplication(input_files, output_dir):
    line_counts = {}
    for path in input_files:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                hash_val = hash(line)
                if hash_val not in line_counts:
                    line_counts[hash_val] = 0
                line_counts[hash_val] += 1

    output_dir_path = Path(output_dir)
    for path in input_files:
        file_name = Path(path).name
        output_path = output_dir_path / file_name
        with open(path, 'r') as f:
            with open(output_path, 'w') as f_out:
                for line in f:
                    line = line.strip()
                    hash_val = hash(line)
                    if line_counts[hash_val] == 1:
                        f_out.write(line)
                        f_out.write('\n')

def minhash_signatures(input_files, num_hashes, n_gram_length):
    all_signatures = []
    for input_file_path in input_files:
        
        signature = []
        with open(input_file_path, 'r') as f:
            doc = f.read()
            doc = normalize_text(doc)
            
            for seed in range(num_hashes):
                min_hash = None 
                
                doc_n_grams = get_n_grams(doc, n_gram_length)
                for n_gram in doc_n_grams:
                    hash_val = mmh3.hash(n_gram, seed)
                    if min_hash is None:
                        min_hash = hash_val
                    else:
                        min_hash = min(min_hash, hash_val)
            signature.append(min_hash)
        all_signatures.append(signature)
    return all_signatures

def minhash_lsh_deduplication(input_files, num_hashes, num_bands, n_gram_length, threshold, output_dir):
    all_signatures = minhash_signatures(input_files, num_hashes, n_gram_length)
    band_size = num_hashes // num_bands
    all_banded_signatures = []
    for signature in all_signatures:
        banded_signature = [signature[j:j+band_size] for j in range(0, len(signature), band_size)]
        all_banded_signatures.append(banded_signature)

    #Grouping documents which have atleast one band in common 
    potential_duplicates = {}
    for first_file_index, first_banded_signature in enumerate(all_banded_signatures):
        for band_index, _ in enumerate(first_banded_signature):
            for second_file_index, second_banded_signature in enumerate(all_banded_signatures):
                if first_file_index == second_file_index:
                    continue
                if first_file_index in potential_duplicates:
                    if second_file_index in potential_duplicates[first_file_index]:
                        continue
                if first_file_index not in potential_duplicates:
                    potential_duplicates[first_file_index] = [] 

                if first_banded_signature[band_index] == second_banded_signature[band_index]:    
                    potential_duplicates[first_file_index].append(second_file_index)

    

    #Evaluate potential duplicates and keep only actual duplicates in the dictionary
    actual_duplicates = {}
    for key, value in potential_duplicates.items():
        first_banded_signature = all_banded_signatures[key]

        if key not in actual_duplicates:
            actual_duplicates[key] = []


        for second_file_index in value:
            second_banded_signature = all_banded_signatures[second_file_index]
            #jackard_similarity = compute_jaccard_similarity(first_banded_signature, second_banded_signature)
            jackard_similarity = compute_jaccard_similarity(key, second_file_index, input_files, n_gram_length)
            if jackard_similarity > threshold:    
                actual_duplicates[key].append(second_file_index)

    #Grouping actual duplicates
    all_duplicate_clusters = get_duplicate_clusters(actual_duplicates)
    
    #Writing the duplicate clusters to output files
    output_dir_path = Path(output_dir)
    for cluster in all_duplicate_clusters:
        random_file_index = random.choice(cluster)
        
        input_path = input_files[random_file_index]
        file_name = Path(input_path).name
        output_path = output_dir_path / file_name
        with open(input_path, 'r') as f:
            with open(output_path, 'w') as f_out:
                for line in f:
                    f_out.write(line)

def compute_jaccard_similarity(first_file_index, second_file_index, input_files, n_gram_length):
    # same_entries = 0
    # for i in range(len(first_banded_signature)):
    #     if first_banded_signature[i] == second_banded_signature[i]:
    #         same_entries += 1
    #     # else:
    #     #     import pdb; pdb.set_trace()
    # return same_entries / len(first_banded_signature)
    doc1 = open(input_files[first_file_index], 'r').read()
    doc2 = open(input_files[second_file_index], 'r').read()

    doc1 = normalize_text(doc1)
    doc2 = normalize_text(doc2)

    doc1_n_grams = get_n_grams(doc1, n_gram_length)
    doc2_n_grams = get_n_grams(doc2, n_gram_length)

    set_doc1_n_grams = set(doc1_n_grams)
    set_doc2_n_grams = set(doc2_n_grams)

    intersection = set_doc1_n_grams.intersection(set_doc2_n_grams)
    union = set_doc1_n_grams.union(set_doc2_n_grams)

    return len(intersection) / len(union)

def get_duplicate_clusters(input_dict):
    all_duplicate_clusters = set()
    for key, value in input_dict.items():
        list_of_keys_checked = [key]
        cluster = set()
        cluster.add(key)
        add_to_cluster(cluster, value, input_dict, list_of_keys_checked)
        cluster = tuple(cluster)
        all_duplicate_clusters.add(cluster)
    return all_duplicate_clusters


def add_to_cluster(cluster, value, input_dict, list_of_keys_checked):
    for val in value:
        if val not in list_of_keys_checked:
            cluster.add(val)
            list_of_keys_checked.append(val)
            add_to_cluster(cluster, input_dict[val], input_dict, list_of_keys_checked)

        
def normalize_text(text):
    # remove accents and NFD unicode normalization
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    # convert to lowercase
    text = text.lower()
    # remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()   
    text = text.replace('\n', ' ')  
    return text      


def get_n_grams(text, n):
    words = text.split()
    n_grams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    return n_grams


def train_quality_classifier():
    model = fasttext.train_supervised(input="quality_classifier_train.txt")
    model.save_model("quality_classifier_model.bin")

def predict_quality_classifier(text):
    model = fasttext.load_model("quality_classifier_model.bin")
    text = text.replace('\n', ' ')
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = normalize_text(text)
    text = text.replace('\n', ' ')
    prediction =  model.predict(text)
    language = prediction[0][0].replace('__label__', '')
    confidence = prediction[1][0]
    return language, confidence
    
def process_single_warc_file(warc_file_path : str, output_file_path : str):
    num_removed_gopher = 0
    num_removed_language = 0
    num_removed_nsfw = 0
    num_removed_toxic = 0
    num_removed_quality = 0

    lang_model = fasttext.load_model('../../lid.176.bin')
    nsfw_model = fasttext.load_model('../../nsfw_model.bin')
    toxic_model = fasttext.load_model('../../hatespeech_model.bin')
    quality_model = fasttext.load_model("quality_classifier_model.bin")

    with open(output_file_path, 'w') as f:
        for record in ArchiveIterator(open(warc_file_path, 'rb')):
            if (record.record_type == WarcRecordType.response):
                if (record.http_headers.get('Content-Type') and 'text/html' in record.http_headers.get('Content-Type')):
                    record_content = record.reader.read()
                    text = extract_text(record_content)

                    text, _ = mask_email(text)
                    text, _ = mask_phone_numbers(text)
                    text, _ = mask_ips(text)
                    #gopher quality filter
                    if(gopher_quality_filter(text) == False):
                        num_removed_gopher += 1
                        continue

                    text = text.replace('\n', ' ')
                    
                    #language detection
                    lang_prediction =  lang_model.predict(text)
                    lang = lang_prediction[0][0].replace('__label__', '')
                    if(lang != 'en'):
                        num_removed_language += 1
                        continue


                    text = re.sub(r'[^\x00-\x7F]+', '', text)
                    text = normalize_text(text)

                    #nsfw classification
                    nsfw_prediction = nsfw_model.predict(text)
                    nsfw_label = nsfw_prediction[0][0].replace('__label__', '')
                    if(nsfw_label != 'non-nsfw'):
                        num_removed_nsfw += 1
                        continue

                    #hatred speech classification
                    toxic_prediction = toxic_model.predict(text)
                    toxic_label = toxic_prediction[0][0].replace('__label__', '')
                    if(toxic_label != 'non-toxic'):
                        num_removed_toxic += 1
                        continue  

                    #quality classification
                    quality_prediction = quality_model.predict(text)
                    quality_label = quality_prediction[0][0].replace('__label__', '')
                    if(quality_label != 'good'):
                        num_removed_quality += 1
                        continue

                    f.write(text)
                    f.write('\n')

    return (output_file_path, num_removed_gopher, num_removed_language, num_removed_nsfw, num_removed_toxic, num_removed_quality)
    
def process_warc_files():
    total_num_removed_gopher = 0
    total_num_removed_language = 0
    total_num_removed_nsfw = 0
    total_num_removed_toxic = 0
    total_num_removed_quality = 0
    executor = submitit.AutoExecutor(folder='slurm_logs')
    max_simultaneous_jobs = 16
    warc_directory_path = Path('/home/shared/CC-MAIN-2023-50-warc-filtered')
    warc_filenames = [f for f in os.listdir(warc_directory_path) if f.endswith('.warc.filtered.gz')]
    output_directory_path = Path('/home/c-akshit/assn4_new1/spring2024-assignment4-data/warc_filtered_data')
    executor.update_parameters(
        slurm_array_parallelism=max_simultaneous_jobs,
        timeout_min=60,
        mem_gb=16,
        cpus_per_task=4,
        slurm_account='cs336_user',
        slurm_partition='a4-cpu',
    ) 
    futures = []
    with executor.batch():
        for warc_filename in warc_filenames:
            warc_filepath = str('/home/shared/CC-MAIN-2023-50-warc-filtered/' + warc_filename)
            future = executor.submit(
                process_single_warc_file,
                warc_filepath,
                os.path.join(output_directory_path, warc_filename.replace('.warc.filtered.gz', '.txt'))
            )
            futures.append(future)
    
    for future in tqdm(
        submitit.helpers.as_completed(futures),
        total=len(warc_filenames),
    ):
        output_file_path, num_removed_gopher, num_removed_language, num_removed_nsfw, num_removed_toxic, num_removed_quality = future.result()
        total_num_removed_gopher += num_removed_gopher
        total_num_removed_language += num_removed_language
        total_num_removed_nsfw += num_removed_nsfw
        total_num_removed_toxic += num_removed_toxic
        total_num_removed_quality += num_removed_quality
        print('Total number of documents removed by gopher filter: ', total_num_removed_gopher)
        print('Total number of documents removed by language filter: ', total_num_removed_language)
        print('Total number of documents removed by nsfw filter: ', total_num_removed_nsfw)
        print('Total number of documents removed by toxic filter: ', total_num_removed_toxic)
        print('Total number of documents removed by quality filter: ', total_num_removed_quality)
        print(f'Output written to {output_file_path}')


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

    #reservoir_sampling('/home/shared/enwiki-20240420-extracted_urls.txt.gz', 100000, 'positive_url_sample1.txt', 'positive_url_sample2.txt')

    # dict = {1: [2], 2 : [1, 3, 4, 5], 3:[2], 4:[5, 6, 7], 5:[2, 4, 8], 6:[4], 7:[4], 8:[5], 9:[10], 10:[9, 11], 11:[10], 12:[]}
    # all_duplicate_clusters = get_duplicate_clusters(dict)
    # print(all_duplicate_clusters)

    #extract_text_from_warc('../../CC-MAIN-20180420081400-20180420101400-00118.warc.gz', 'negative_urls_text.txt')

    #train_quality_classifier()
    process_warc_files()
    return 0


if __name__ == '__main__':
    main()