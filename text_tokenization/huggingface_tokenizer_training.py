from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from collections import Counter
import re

# Eğitim verisini yükleyin
with open('NUTUK_updated2.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Kelimeleri ayıklayın
words = re.findall(r'\w+', text.lower())

# Kelime frekanslarını hesaplayın
word_freq = Counter(words)

# Benzersiz kelime sayısını bulun
unique_words = len(word_freq)

print(f"Benzersiz kelime sayısı: {unique_words}")

# En sık kullanılan kelimelerin frekanslarını görüntüleyin
print("En sık kullanılan kelimeler:", word_freq.most_common(10))

# Özel bir BPE modeli oluşturun
tokenizer = Tokenizer(models.BPE())

# Tokenizer'a eğitim verisini tanıtın
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Tokenizer'ınızın belirli tokenları tek parça olarak algılamasını istediğiniz kelimeleri belirleyin
special_tokens = ["<k>", "<e>", "<location>", "<img>", "<os_img>", "<deleted>"]

# Trainer'ı ayarlayın ve eğitin
trainer = trainers.BpeTrainer(vocab_size=20000) #, special_tokens=special_tokens
tokenizer.train_from_iterator([text], trainer=trainer)

# Tokenizer'ı kaydedin
tokenizer.save("Nutuk_tokenizer.json")