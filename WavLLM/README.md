# WavLLM

## Model
to get the WavLLM model, run
```bash
bash ./download/download.sh 0
```

## Gaokao (SQA)
The audio samples and transcripts can be download using
```bash
bash ./download/download.sh 1
```

The tsv file path of SQA task is [`tsv_path`](wavllm/test_data/gaokao.tsv).

## Setup

```bash
git submodule update --init WavLLM/fairseq
cd WavLLM/
conda create -n wavllm python=3.10.0
conda activate wavllm
pip install --editable fairseq/
pip install sentencepiece
pip install transformers==4.32.1
pip install numpy==1.23.5
pip install editdistance
pip install soundfile
```

## Inference
```bash
cp -r wavllm fairseq/examples
cd fairseq
bash examples/wavllm/scripts/inference_sft.sh $model_path $data_name
```
We provided examples of each task in [`test_data`](wavllm/test_data)

## Examples
| Task |  Audio | Prompt |  Target |  Output |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| ASR | [`audio`](wavllm/test_data/audio/asr.flac) | Based on the attached audio, generate a comprehensive text transcription of the spoken content. | he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce | he hoped there would be stew for dinner turnips and carrots and browned potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce  |
| SV | [`audio`](wavllm/test_data/audio/sv.wav) | Is there only one speaker in the audio clip? | Yes | Yes, there is only one speaker in the audio clip  |
| ST | [`audio`](wavllm/test_data/audio/st.flac) | Translate the audio clip into German. | Sie wird schon in Ordnung sein. | Sie wird es wohl überleben.  |
| ER | [`audio`](wavllm/test_data/audio/emo.wav) | Can you describe the emotional condition of the speaker in the provided audio clip? | sad | sad  |
| SQA | [`audio`](wavllm/test_data/audio/sqa.wav) | What will the man do next? A. Start to take exercise; B. Do as he always does; C. Change his working time. | A | A |
| SQQA | [`audio`](wavllm/test_data/audio/sqqa.wav) | - | The fundamental theorem of calculus is a theorem that links the concept of the derivative of a function with the concept of the integral. | The fundamental theorem of calculus states that different rates of change of a function at a point in the domain of the function are equal. This theorem is used to determine the accumulation of a quantity over a given interval, and to establish relationships between different variables. It is also used to calculate derivatives, which are important in many areas such as optimization, physics, and economics. |
| II-task | [`audio`](wavllm/test_data/audio/II-task.wav) | To begin, Transcribe the audio recording into text, capturing every spoken word; Subsequently, How does the woman finally decide to go home? A. By bus; B. In the man’s car; C. In her father’s car.; Furthermore, ignore the audio clip, What is the capital of New Zealand?; Lastly, Continue the narrative of given audio clip in a coherent and engaging way |  - | Here's the text from the audio recording: o no i missed the bus home it's already half past ten now i don't think there would be another bus to come don't worry i can take you home thanks but we go in different directions i'll just call my dad to pick me up\\nAddressing the prompt 'How does the woman finally decide to go home? A. By bus; B. In the man’s car; C. In her father’s car.', the answer is: C\\nThe response to 'What is the capital of New Zealand? is: The capital of New Zealand is Wellington.\\nContinuing the story, we have: The woman looked at the man with gratitude and relief. She had been worried about how she would get home after missing the bus. The man smiled and said, “It’s no problem, I can take you home.” The woman thanked him and they made their way out of the building. As they walked, the woman realized that they were going in different directions. She thanked the man again and promised to call her dad to pick her up. As they parted ways, the man wished her a good night and she walked towards her dad’s house, feeling grateful for the kindness of a stranger.  |
| CoT-task | [`audio`](wavllm/test_data/audio/CoT-task.wav) | First of all, transcribe the audio recording into text, capturing every spoken word; Additionally given this audio clip and text, can you condense it into a clear, concise summary, no more than 20 words?; Lastly disregarding the sound, translate this English summary into German. | Drei Filme aus dem asiatisch-pazifischen Raum im Rennen in Cannes | The audio transcription is: three films from asia pacific are in the running for the coveted golden palms at this year's cannes film festival competing in a field dominated by european productions organizers announced monday; The audio clip have been summarized as follows: Three films from Asia-Pacific are competing for Golden Palms at this year's Cannes Film Festival, challenging European productions.; The translation of summary is: **Drei Filme aus Asien-Pazifik nehmen an diesem Jahr an den Filmfestspielen von Cannes teil und konkurrieren mit europäischen Produktionen um die Goldenen Palmen.**  |
| CoT-task-story | [`audio`](wavllm/test_data/audio/CoT-task-story.wav) | First of all, transcribe the audio recording into text, capturing every spoken word; Additionally given this audio clip and text, can you condense it into a clear, concise summary, no more than 20 words?; Lastly disregarding the sound, translate this English summary into German. | Bis zum Jahr 2500 ist die Erde eine umweltfreundliche Utopie mit fortschrittlicher KI, neuronaler Vernetzung und einer perfekten Mischung aus Technologie und Natur. | The audio clip's transcription reads: in the year twenty five hundred earth gleamed like a sapphire a futuristic utopia where harmony reigned skyscrapers draped in lush greenery stretched towards the heavens their glass surfaces reflecting the tranquil azure of a pollution free sky humanity had transcended past conflicts embracing an era of shared consciousness through neural connectivity autonomous vehicles glided silently on solar paths while people mingled in serene communal spaces their basic needs met by advanced ai that predicted and catered to their every whim the great reconciliation had merged technology with nature and in this new world every individual thrived their potential limited only by the expanse of their own creativity the utopia wasn't just a place it was the pulse of civilization beating in perfect rhythm with the universe; The summary of the audio clip is: In 2500, Earth became a futuristic utopia with harmonious, pollution-free society, advanced technology, and limitless individual potential.; The summary translates to: **Im Jahr 2500 wurde die Erde zu einer futuristischen Utopie mit harmonischer Gesellschaft, reiner Umwelt, fortschrittlicher Technologie und unbegrenztem menschlichem Potenzial.** |
