# YouTube commenter

## Links 
* Data: [YouTube comments](https://www.kaggle.com/tanmay111/youtube-comments-sentiment-analysis)
* [Data Preprocess](https://www.kaggle.com/code/tanmay111/youtube-comments-sentiment-analysis/notebook) 
* Model: [GPT-2 large](https://huggingface.co/gpt2-large)
* [GPT-2 Finetune example](https://tinyurl.com/gpt2-finetune-colab)
* Generation [example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py)
* [Decoding methods](https://huggingface.co/blog/how-to-generate)

## Technical requirements 
* В итоговом сервисе можно дать пользователю вариировать параметры генерации: температура или top-p, если сэмплинг; beam size и length penalty, если beam search; сколько комментариев сгенерировать, etc. 
* Отдельный респект если ваш код будет выводить комментарий по одному слову, прямо в процессе генерёжки - чтобы пользователь не ждал пока вы настругаете абзац целиком.

## Fine-tune
* Dataload and preprocessing - dataset.py  
* Trainer script - trainer.py
* Training time - 5h (trainer), 87h (custom_train)
* GPU - Nvidia-GTX 2070
* Experimets:
    * Stupid - without exra symbols + lower case  
    * Smart - with exra symbols + origin case 
* Results
    | Exp    | TrainLoss | TestLoss | Perplexity |
    |--------|-----------|----------|------------|
    | Stupid | 2.128     | 2.133    | 8.44       |
    | Smart  | 2.362     | 2.335    | 10.23      |

## Run streamlit

``` CUDA_VISIBLE_DEVICES="1" streamlit run streamlit_core.py ```

Example link: `https://www.youtube.com/watch?v=CUwg_JoNHpo`


<img src="https://github.com/MaximovaIrina/youtube_commenter/blob/main/data/demo.gif" width="450" height="480">
