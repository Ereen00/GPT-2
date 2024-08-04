Orijinal GPT-2 modeli değiştirilerek ve türkçe metinleri de işleyebilmesi sağlanarak değiştirildi. 
Proje Data_cleaning, Tokenizing ve Main_model olarak üçe ayrılıyor. 
Model ile daha önce Nutuk metni ile eğitildi fakat şu anda içerisinde hiç bir dataseti bulunmuyor. Kullanıcılar kendi datasetlerini aşağıdaki adımlardan geçirerek modeli eğitip kullanabilirler.

-----------------------------------------------------------------------------

=> data_cleaning bölümünde kaba halde bulunan data yazım hatalarından ve büyük harflerden ayıklanır. Özel token'lar daha sonra tokenizer'da eğitilmek üzere bu bölümde oluşturulur.

=> Her data için temizlik süreci farklı olduğundan bu bölümün içi boş bırakıldı.

------------------------------------------------------------------------------

=> SPM: text_tokenization bölümünde temizlenen metin metine özel bir şekilde tekrar eden kelime gruplarına bakılarak tokenize edelir. Önceki bölümde eklenen tokenlar bu bölümde user_defined_symbols adı altında etiketlenir, böylece özel tokenlar alt parçalara bölünmekten kurtarılmış olunur. tokenized_output isminde temizlenmiş metinin tokenlara bölünmüş txt formatındaki hali, tokenizer.model isminde modelde kullanılmaya hazır hali ve tokenizer isminde her bir token ve parametresi sözlük şeklinde biçimlendirilmiş txt formatındaki hali oluşturulur.

=> HUGGINGFACE: text_tokenization bölümünde temizlenen metin metine özel bir şekilde tekrar eden kelime gruplarına bakılarak tokenize edelir. special_tokens adı altında toplanılan kelime veya hece grupları tek birer token olarak gösterilebiliyor (kullanımında bazı sorunlar mevcut) tokenizer.json adında modelde kullanılmaya hazır bir JSON dosyası oluşturulur.

------------------------------------------------------------------------------

=> Model eğitim sırasında kaydettiği değerlerin grafiklerini Training_process dosyasına, eğitilmiş modelin kendisini ise log dosyasına kaydeder.
