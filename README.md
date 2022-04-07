# YouTube commenter

* Data: [YouTube comments](https://www.kaggle.com/tanmay111/youtube-comments-sentiment-analysis)
* Data [preprocess](https://www.kaggle.com/code/tanmay111/youtube-comments-sentiment-analysis/notebook) 
* Model: [GPT-2 large](https://huggingface.co/gpt2-large)
* [GPT-2 Finetune example](https://tinyurl.com/gpt2-finetune-colab)

## Technical requirements 
* В итоговом сервисе можно дать пользователю вариировать параметры генерации: температура или top-p, если сэмплинг; beam size и length penalty, если beam search; сколько комментариев сгенерировать, etc. 
* Отдельный респект если ваш код будет выводить комментарий по одному слову, прямо в процессе генерёжки - чтобы пользователь не ждал пока вы настругаете абзац целиком.

## Status
* Разбить на тест и вал датасет [YouTube comments](https://www.kaggle.com/tanmay111/youtube-comments-sentiment-analysis)
 
