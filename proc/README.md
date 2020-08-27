# This repo is used for data pre-processing for ACE event


### [ACE05](https://catalog.ldc.upenn.edu/LDC2006T06) Event

#### Creating the dataset

The preprocessing code I wrote breaks with the newest version of Spacy. So unfortunately, we need to create a separate virtualenv that uses an old version of Spacy and use that for preprocessing.

```shell
conda deactivate
conda create --name ace-event-preprocess python=3.7
conda activate ace-event-preprocess
pip install -r scripts/data/ace-event/requirements.txt
python -m spacy download en
```

Then, collect the relevant files from the ACE data distribution with

```
bash ./scripts/data/ace-event/collect_ace_event.sh [path-to-ACE-data].
```

The results will go in `./data/ace-event/raw-data`.

Now, run the script

```
python ./scripts/data/ace-event/parse_ace_event.py [output-name] [optional-flags]
```

You can see the available flags by calling `parse_ace_event.py -h`. For detailed descriptions, see [DATA.md](DATA.md). The results will go in `./data/ace-event/processed-data/[output-name]`. We require an output name because you may want to preprocess the ACE data multiple times using different flags. For default preprocessing settings, you could do:

```
python ./scripts/data/ace-event/parse_ace_event.py default-settings
```

Run convert script to convert to one sentence/line format:

```
python scripts/data/ace-event/convert_examples.py
```

When finished, you should `conda deactivate` the `ace-event-preprocess` environment and re-activate your modeling environment.


### Adaptation

The scripts are adapted with the [Dygiepp repo](https://github.com/dwadden/dygiepp).