# Отчет по БДЗ#1 от Ершова Ивана (tutugarin) 

В рамках данного домашнего задания нужно было обучить модель для генерации рассказов и сравнить полученную модель с `GPT2-XL`

## Список файлов:

В этом разделе описаны файлы, которые будут представлены для сдачи, и краткое описание каждого из файла. Подробное описание каждого из файла будет далее.

1. `dataset.py` -- токенайзер, датасет, парсинг данных, даталоадер, полезные функции для работы с данными
1. `model.py` -- опеределение модели
1. `train.py` -- функции для обучения и валидации
1. `utils.py` -- некоторые утилити функции и конфигурация обучения
1. `main.py` -- скрипт для запуска обучения
1. `inference.py` -- скрипт для удобной генерации текста
1. `checkpoint_best.pt` -- лучшая модель, здесь сохранена модель, оптимизатор, лучший лосс, списки из тренировочных и валидационных лосов
1. `bpe.model` `bpe.vocab` -- файлы для токенизатора
1. `training_logs` -- все логи обучения


## Описание содержания файлов:

Краткое описание имеющихся функций и классов для каждого файла

### `dataset.py`

Обработка данных

- `Tokenizer` -- токенайзер, который на вход принимает `.txt` файл со всеми текстами (формат файла: на каждой строчке новый рассказ). Если до этого не был создан токенизатор, то создается `SentencePieceProcessor`, иначе загружает `SentencePieceProcessor` из файла. Используется `Byte-Pair Encoding tokenization`, размер словаря `5120`. `Tokenizer` так же определяет методы, которые кодируют предложения в последовательности токетонв и декодирует токены в предложения

- `TextDataset` -- датасет, который наследуется от `torch.utils.data.Dataset`. На вход принимается файл со всеми текстами, они токенизируются и сохраняются в поле класса. В методе `__getitem__`, которые принимает на вход индекс, происходит добавление `<bos>` и `<eos>` токенов. Также, если длина исходной последовательности токенов превышает `max_lex - 2`, то последовательность обезается до `max_lex - 2` токенов. Длинных предложений в исходном датасете немного, поэтому такой подход некритично мешает обучению, но при этом позволяет больше данных грузить в батч

- `parse_jsons` -- парсит джоны и сохраняет в один файл. На вход принимает конфиг, в котором указана папка с `.json` файлами, и принимает название выходного файла, куда записать результаты парсинга.

- `get_dataloaders` -- делает следующие вещи:
    - проверяет, если ли файл со всеми тексами. Если нет, то вызывает `parse_jsons`
    - создает `Tokenizer`
    - создает `TextDataset`
    - делит `TextDataset` на два датасета: тренировочный и валидационный в отношении `0.995 : 0.005` соответсвенно. Таким образом в валидации `~24000` историй
    - возвращает даталоадеры для тренировки, валидации и возвращает токенизатор


### `model.py`

Архитектура модели

- `PositionalEncoding` не нужндается в представлении

- `LanguageModel` -- encoder-based модель. Создает `PositionalEncoding`, трансформер `torch.nn.TransformerEncoder`, который принимает в себя `torch.nn.TransformerEncoderLayer` и `torch.n.LayerNorm`, и голова `torch.nn.Linear`, предсказывающие логиты. 
    - инициализация параметров распределением Ксавьера
    - используемые гиперпараметры:
        - embed_dim: 512
        - hidden_dim: 2048
        - num_heads: 8
        - num_layers: 6
        - dropout: 0.1
    - модель реализует логику генерации текста в методе `inference`, подробнее о генерации текста будет рассказано в отдельном пункте


### `train.py`

Реализованы функции для обучения и валидации

- `_validate` -- проходит по всему валидационному сету и считает лосс
- `validate_step` -- вызывает `_validate` и после генерирует пробный текст, начиная с фразы `"Once upon a time "`. Так же делает замер скорости генерации
- `train_step` -- делает 1 шаг оптимизации. Здесь находится логика применения AMP и grad clipping
- `train_epoch` -- проходится по всем тренировочным данным. Вызывает `validate_step` раз в 1000 итераций, сохранят `checkpoint_best.pt` и `checkpoint_last.pt`. `checkpoint_best.pt` определяется по лоссу на валидационной выборке. После подсчета ошибки на валидационной выборке, значение этой ошибки передается в `torch.optim.lr_scheduler.ReduceLROnPlateau`
- `train` -- вызывает количество эпох раз `train_epoch` 


### `utils.py`

- `get_logger` -- создает логгер из библиотеки `logging`
- `save_model` -- сохраняет состояние модели и оптимизатора. Сохраняет последний лосс модели. Так же сохраняет два списка тренировочных и валидационных лоссов.
- `Config` -- простенький `dataclass`, в который записаны некоторые константы и гиперпараметры


### `main.py`
Главный скрипт обучения. Пример запуска: 

```bash
CUDA_DEVICE=7 python3 main.py
```

- берет индекс `GPU` из переменной окружения `CUDA_DEVICE`
- создает модель `LanguageModel`
- создает оптимизиатор `torch.optim.Adam`
- загружает состояние модели и оптимизатора из `checkpoint_last.pt`, если такой файл существует
- создает lr шедулер `optim.lr_scheduler.ReduceLROnPlateau`
- создает `torch.cuda.amp.GradScaler` для работы с `AMP`
- вызывает функцию `train` из `train.py`


### `inference.py`

Скрипт для генерации текстов. Пример запуска:

```bash
CUDA_DEVICE=7 python3 inference.py \
    --checkpoint-path "checkpoint_best.pt" \
    --tokenizer-path "bpe.model" \
    --gen-topk 2 \
    --gen-temperature 0.5 \
    --gen-quantity 32 \
    --output-file gen_4.txt \
    --promt "Once upon a time "
```

- `--promt` -- начальный текст истории, который модель будет продолжать
- `--checkpoint-path` -- путь до модели, сохранненной в том же формате, что и во время обучения
-  `--tokenizer-path` -- путь до обученного токенайзера
- `--output-file` -- путь до файла, куда будут записываться сгенерированные тексты. Если не указать этот параметр, то сгенерированные тексты будут выводиться в `sys.stdout`
- `--gen-temperature` -- число, на которое будут делиться логиты выхода модели, влияет на генерацию
- `--gen-topk` -- число токенов, из которых будет семплироваться следующий
- `--gen-quantity` -- число текстов, которое надо сгенерировать


### `checkpoint_best.pt`

Сохраненный чекпоинт модели. Представляет из себя словарь с полями:
- `model` -- состояние модели
- `optim` -- состояние оптимизатора
- `loss`
- `train_losses` -- список тренировочных лоссов
- `val_losses` -- список тренировочных лоссов


## Процесс обучения

Обучнение производилось на a100-40gb. Основные особенности обучения:

- обучение с `AMP` и `bf16`
- максимальное число эпох выставлено 100, однако обучение можно прервать в ручную. Поскольку раз в 1000 итераций сохраняется и `checkpoint_last.pt`, и `checkpoint_best.pt`, то всегда есть возможность продолжить обучение
- `batch_size = 512; max_len = 256`, таким образом в одном батче `~130000` токенов
- финальная модель обучалась 11 эпох
- изначальный `lr=3e-4`, используется шедулер `ReduceLROnPlateau`, если 10 валидаций (то есть 10 * 1000 итераций обучения) лосс не улучшался, то `lr *= 0.3`
- после каждой валидации также происходит пробная генерация текста, чтобы отслеживать качество модели

Для запуска обучения нужно предварительно в папку с данными положить и разархивировать данные:
```bash
wget --quiet --show-progress "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
tar -xf TinyStories_all_data.tar.gz -C .
```

P.S. изначально я не сохранял ошибки на тренировочной и валидационной выборке, поэтому графиков с обучением нет. Я добавил сохранение и поставил модель обучатсья заново, но есть вероятность, что я утром просплю дедлайн и не добавлю графики.


## Генерация

Код генератора находится в методе `inference` у модели. Этот метод принимает на вход токенизатор, промт и максимальное число токенов для генерации. Так же есть два параметра, влияющие на генерацию: температура и некоторое число `topk_tokens`, про которое я далее расскажу на примере. Как происходит генерация:

- промт переводится в токены, кладутся в список `context_tokens`
- запускается цикл до тех пор, пока не будет предсказан `<eos>` или пока не будет сгенерировано максимальное число токеновЖ
    1. `context_tokens` передаются в модель, модель возвращает вероятностное распределение токенов `logits`
    2. выбирается `topk_tokens` наиболее вероятных токенов из `logits`
    3. `logits` делятся на температуру
    4. полученные `logits` интерпретируются как веса токенов. `logits` передаеются в мультимодальное распределение, и чем выше вес, тем более вероятно будет выбран токен 
    5. семплируется новый токен, он кладется в `context_tokens`
    6. цикл повторяется

Замечания:
- если `topk_tokens == 1`, то получаем просто генерацию наиболее вероятных токенов
- чем выше `topk_tokens` тем более разнообразные могут быт истории
- температура варьируется от 0 до 1. Чем ближе к 0, тем равновероятно будем выбирать следующий токен из `topk_tokens` токенов, что увеличивает разноообразность генерации
- если делать очень разнообразную генерацию (большй `topk_tokens`, маленькая температура), то будет получаться ерунда


## Эксперименты и сравнение с GPT2-XL

- GPT2-XL 1.5B параметров
- наша модель 24m параметров

Посмотрим на различные генерации текстов и сделаем выводы:

- `temperature=0.1`, `topk_tokens=3`:
```
"""
Once upon a time there was a little bird named bob, who was always very curious and wanted to know what the universe had to offer him, but it was too big for his tiny beak to understand the big world of the stars in the night. one day he decided to take a chance. as bob flew through space, he noticed something shiny and beautiful on his way. it was an old, fragile egg that was lying near a tree, and bob was curious about it, so when the egg started to crack open and inside he found a tiny, fragile egg that was made from gold. "what a special surprise!" said the egg as bob looked around to see who it belonged to, but no matter how many days he flew around with it. finally he decided that the egg was his and he was so proud to have discovered it, he decided that it was his new home. bob flew around showing his friends his new home, and they all thought it was amazing too. bob was very happy that he decided that the fragile little egg had brought such an amazing surprise that he had never heard about before. he knew it was the best decision to have found and that the egg was now his home. from now, the little fragile bird and the tiny, the end the end the
"""
```

(специально подобранный плохой пример) Полученный текст не очень имеет полноценного смысла, неодушевленные объекты начинают говорить. В конце видим начало галлюцинаций (повторение "the end"), генерация остановилась из-за достижения максимального числа токенов

- `temperature=1`, `topk_tokens=3`:
```
"""
Once upon a time in an ordinary house lived two best friends, jack. jack was always so happy, he would laugh and smile every time. one morning, jack woke and decided to go outside and look around. he saw a beautiful rainbow and wanted to touch the sky, so he started walking towards it, hoping to find something to touch. as jack reached the rainbow in the sky he noticed a small bird. it looked so sad, so it flew up to the sky and started chirping away, so he could see it better. the bird was so happy, it started to sing. suddenly, jack heard the bird's voice and he was amazed! he had never seen such an ordinary bird before and he wanted to know why the bird't come down from the sky, but he couldn t figure that out. he asked the other bird why it was singing so loud. the bird replied that the bird was trying to find a new home and that jack could help the bird. jack smiled. "i'm glad you'd found something to help you," the bird replied. jack thanked the two friends and went on to help their bird find the new nest. they looked around and soon they saw it was a cozy nest with a cozy nest inside and the from
"""
```
В начале есть какая-то история, некоторая связанная сюжетная линия. Но в конце происходит непонятное, диалоги персонажей становятся несвязанными

- `temperature=1`, `topk_tokens=2`:
```
"""
Once upon a time there were three friends. they were called tom, sam and jack. tom had a special toy. he was very proud of his special object. he showed it to all his best friends and said, look at my special toy. i am proud to show it off! sam and lily were jealous. they wanted to see the object. tom said, let's go and see what it can be! so, tom and lily ran off. when they got to the top, they were surprised to find out that the toy had a special surprise. it was an amazing toy! tom said, let s play with this toy! we can all play with it together. sam, sam, and jack were so excited to have the new object. they played together for hours, laughing, having lots of adventures with their new toy. they all agreed that it was a very fun and special toy!
"""
```
пу-пу-пу

- `temperature=0.5`, `topk_tokens=2`:
```
"""
Once upon a time , in a big green forest with lots of animals. one day, a little rabbit was hopping along when it heard a loud noise. the noise was coming from a nearby tree. it sounded like a bird, but the bird was singing a song that made it feel happy and peaceful. suddenly the rabbit saw a big, scary bear coming out of the forest. it started to run as fast and the bear chased after it. but the little bunny was too  ⁇ uickly and the bird couldn't catch it. just when the bear was about to catch it, a kind deer came along and scared it away. the little rabbit was safe, and the bear never came back. from that moment on, they became the best friends.
"""
```
Хороший добрый рассказ, модель не путается в персонажах. Есть один артефакт, когда модель предсказала ⁇, но можем считать это погрешностью. 

- примеры генерации GPT2-XL
```
"""
Once upon a time, the Federal Trade Commission (FTC)--the nation's consumer protection agency that was created, in part, to protect consumers from unfair or deceptive business practices--received requests for information about the nature of Google's business, such as searches (for example), and Google's search engine itself.This form of inquiry was intended to gather more information about the extent of Google's business and the practices that it is engaging in. The questions posed to Google concerned its methods for collecting
"""
```
GPT2 пытается расскзывать какие-то факты. Что примечательно, что промт почти не повлиял на рассказ. Видимо, что генерация текста остановилась из-за максимального числа токенов 
```
"""
Once upon a time, the name "the Black Lives Matter movement" would have seemed so alien to the American imagination that the phrase would have been regarded as completely innocuous, or even humorous. But in the past decade, an organized effort from Black youth has made the movement a lightning rod for widespread protests around the country from Ferguson, Mo. to Baltimore, Md. — and one that even Republican presidential candidates have picked up on.( Also on POLITICO: Who's talking about Black lives?
"""
```
Опять какой-то факт, что текст похож на какую кашу. Совсем не похоже на историю. Еще стоит заметить артефакт предсказания "(". 

Если делать визуальную оценку сгенерированных текстов, то GPT плохо справляется в сочинение историй. Она знает много фактов и пытается их рассказать, но не в формате историй. **Наша модель определенно выигрывает в сочинение**.

Это была некоторая субъективная оценка, теперь попробуем ввести более объективную метрику. Будем оценивать качества генерации с помощью perplexity, для этого возьмем большую и хорошо обученную языковую модель, которая училась на огромном корпусе текстов. Считается, что большие языковые модели хорошо выучивают естественный язык, потому с их помощью мы сможем оценивать качество наших маленьких моделей. 

Сгенерируем тексты при помощи нашей модели и при помощи GPT2, дадим их на вход большой модели. Большая модель будет предсказывать токены как во время обучения. Мы полагаем, что большая модель хорошо знает язык, поэтому perplexity на сгенерированных текстах будет нам говорить о "человечности" текста. Чем меньше perplexity, тем лучше, значит текст более похож на человеческий.

В качестве большой модели возьмем [Starling-LM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha): 7B модель для генерации текста, качество которой сопоставимо с GPT4.

Полученные значения perplexity:
- GPT2-XL 22.08
- Наша модель `temperature=0.1`, `topk_tokens=3`: 56.25
- Наша модель `temperature=1`, `topk_tokens=3`: 39.42
- Наша модель `temperature=1`, `topk_tokens=2`: 38.70
- Наша модель `temperature=0.5`, `topk_tokens=2`: 21.37

Получаем, что при параметрах генерации `temperature=0.5`, `topk_tokens=2` получаем качество на нашей модели лучше, чем на GPT2-XL


## Заключение

Был написан фреймворк для обучения моделей генерации текста, написан скрипт для удобной генерации текстов.

Были проведены эксперименты с параметрами генерации (`temperature`, `topk_tokens`), продемонстрированы влияние этих параметров на генерацию.

Было проведено сравнение нашей модели с GPT2-XL. GPT2-XL превосходит нашу модель в знании фактов, однако GPT2-XL для генерации небольших рассказов. Это было и показано наглядно на примерах, и показано при помощи введения метрики оценивания моделей через большую языковую модель