import sentencepiece as spm

# Eğitim verisini hazırlayın
with open('dataset.txt', 'r', encoding='utf-8') as file:
    text = file.read()

with open('dataset_for_training.txt', 'w', encoding='utf-8') as file:
    file.write(text)

# SentencePiece modeli eğitin
spm.SentencePieceTrainer.train(
    input='dataset_for_training.txt', 
    model_prefix='tokenizer',
    vocab_size=10000, 
    user_defined_symbols=['<img>','<deleted>','<os_img>','<e>','<k>','<location>']
    ) 

# Modeli yükleyin
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')

# Test metni
with open('dataset.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Metni tokenize edin
tokens = sp.encode(text, out_type=str)

# Sonuçları yazdır veya başka bir dosyaya kaydet
with open('tokenized_output.txt', 'w', encoding='utf-8') as output_file:
    for token in tokens:
        output_file.write(token + '\n')

print("Tokenization completed. Check the 'tokenized_output.txt' file for results.")
