import dataclasses
import pprint
import time
from functools import partial
import json
import base64
from multiprocessing import Pool, Value, Lock, Manager

import h5py
import mlxu
from ml_collections.config_dict import config_dict
from ml_collections import ConfigDict
from tqdm import tqdm, trange
import numpy as np
import threading

from datasets import load_dataset
from datasets import interleave_datasets

class DatasetFactory(object):
    """ Datset builder class. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.type = 'huggingface'
        config.text_processor_class = 'TextProcessor'
        config.text_processor = globals()[config.text_processor_class].get_default_config()
        config.huggingface_dataset = HuggingfaceDataset.get_default_config()
        config.json_dataset = JsonDataset.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        text_processor = globals()[config.text_processor_class](config.text_processor, tokenizer)
        if config.type == 'huggingface':
            return HuggingfaceDataset(
                config.huggingface_dataset, tokenizer, text_processor, **kwargs
            )
        elif config.type == 'json':
            return JsonDataset(config.json_dataset, tokenizer, text_processor, **kwargs)
        else:
            raise ValueError(f'Unknown dataset type: {config.type}')

    def __init__(self):
        raise ValueError('DatasetFactory is a static class and should not be instantiated.')
    
class TextProcessorMetadata:
    def __init__(self) -> None:
        self.manager = Manager()
        self.tag_index_map = self.manager.dict()
        self.tag_index_reverse_map = self.manager.dict()
        self.tag_index = Value('i', 0)
        self._lock = Lock()

    def get_tag_index_map(self):
        with self._lock:
            return self.tag_index_map
        
    def update_tag_index_map(self, new):
        with self._lock:
            self.tag_index_map.update(new)

    def get_tag_index(self):
        with self._lock:
            return self.tag_index.value
        
    def set_tag_index(self, new):
        with self._lock:
            self.tag_index.value = new

    def get_reverse_tag_index_map(self):
        with self._lock:
            return self.tag_index_reverse_map
        
    def update_reverse_tag_index_map(self, new):
        with self._lock:
            self.tag_index_reverse_map.update(new)
        

_metadata = TextProcessorMetadata()


class TextProcessor(object):
    """ Example processor that converts a dictionary of texts into tokens. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ''
        config.fields = ''
        config.tag = ''
        config.subfield_separator = ' '
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ''
        config.base64_token_dtype = 'i4'
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields != '' or self.config.fields_from_example != '', (
            'Either fields or fields_from_example must be specified.'
        )
        self.tokenizer = tokenizer

    def __call__(self, example, has_aux=False):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()

        tag = example.get(self.config.tag, 'en')
        if tag not in _metadata.get_tag_index_map():
            print('tag: %s has index: %d' % (tag, _metadata.get_tag_index()))
            _metadata.update_tag_index_map({tag: _metadata.get_tag_index()})
            _metadata.update_reverse_tag_index_map({_metadata.get_tag_index(): tag})
            tag_index = _metadata.get_tag_index()
            _metadata.set_tag_index(tag_index + 1)
        else:
            tag_index = _metadata.get_tag_index_map()[tag]
            # print('tag: %s has index: %d, debug:%s, %d' % (tag, tag_index, str(_metadata.get_tag_index_map()), _metadata.get_tag_index()))

        token_buffer = []
        loss_mask_buffer = []

        if self.config.add_bos_token:
            token_buffer.append(self.tokenizer.bos_token_id)
            loss_mask_buffer.append(0.0)

        if self.config.fields_from_example != '':
            fields = example[self.config.fields_from_example].split(',')
        else:
            fields = self.config.fields.split(',')

        for i, field in enumerate(fields):
            if field.startswith('[') and field.endswith(']'):
                # No loss for this field.
                field = field[1:-1]
                mask = 0.0
            else:
                mask = 1.0

            if field.startswith('<|') and field.endswith('|>'):
                # Special tokens.
                field = field[2:-2]
                if field == 'bos':
                    token_buffer.append(self.tokenizer.bos_token_id)
                elif field == 'eos':
                    token_buffer.append(self.tokenizer.eos_token_id)
                else:
                    # Token ID specified directly.
                    token_buffer.append(int(field))
                loss_mask_buffer.append(mask)
            elif field.startswith('{') and field.endswith('}'):
                field = field[1:-1]
                # Base64 encoded raw tokens.
                tokens = np.frombuffer(
                    base64.b64decode(example[field]),
                    dtype=self.config.base64_token_dtype
                ).tolist()
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])
            else:
                subfields = field.split('+')
                text = self.config.subfield_separator.join(
                    [example[subfield] for subfield in subfields if subfield in example and example[subfield]]
                )
                if not text:
                    continue
                if i == 0:
                    text = self.config.prepend_text + text
                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

        if self.config.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)

        return token_buffer, loss_mask_buffer, [tag_index] * len(token_buffer), *aux
    

class InstructSingleChoiceTextProcessor(object):
    """ processor that converts a instruction text format for single choice answer selection into tokens. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.conversation_key = 'conversation'
        config.tag = ''
        config.subfield_separator = ' '
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ''
        config.base64_token_dtype = 'i4'
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.conversation_key != '', (
            'conversation_key must be specified.'
        )
        self.tokenizer = tokenizer
        self.inst_begin_tokens = tokenizer.encode('[INST]', add_special_tokens=False)
        self.inst_end_tokens = tokenizer.encode('[/INST]', add_special_tokens=False)

    def __call__(self, example, has_aux=False):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()

        tag = example.get(self.config.tag, 'en')
        if tag not in _metadata.get_tag_index_map():
            print('tag: %s has index: %d' % (tag, _metadata.get_tag_index()))
            _metadata.update_tag_index_map({tag: _metadata.get_tag_index()})
            _metadata.update_reverse_tag_index_map({_metadata.get_tag_index(): tag})
            tag_index = _metadata.get_tag_index()
            _metadata.set_tag_index(tag_index + 1)
        else:
            tag_index = _metadata.get_tag_index_map()[tag]
            # print('tag: %s has index: %d, debug:%s, %d' % (tag, tag_index, str(_metadata.get_tag_index_map()), _metadata.get_tag_index()))

        token_buffer = [self.tokenizer.bos_token_id]
        loss_mask_buffer = [0]

        conversations = example[self.config.conversation_key]

        for conv in conversations:
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            human_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            assistant_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

            input_tokens = self.inst_begin_tokens + human_tokens + self.inst_end_tokens
            output_tokens = assistant_tokens

            token_buffer += input_tokens + output_tokens
            loss_mask_buffer += [0] * len(input_tokens) + [0, 1, 0]

        return token_buffer, loss_mask_buffer, [tag_index] * len(token_buffer), *aux
    

class InstructTextProcessor(object):
    """ processor that converts a instruction text format into tokens. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.conversation_key = 'conversation'
        config.tag = ''
        config.subfield_separator = ' '
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ''
        config.base64_token_dtype = 'i4'
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.conversation_key != '', (
            'conversation_key must be specified.'
        )
        self.tokenizer = tokenizer
        self.inst_begin_tokens = tokenizer.encode('[INST]', add_special_tokens=False)
        self.inst_end_tokens = tokenizer.encode('[/INST]', add_special_tokens=False)

    def __call__(self, example, has_aux=False):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()

        tag = example.get(self.config.tag, 'en')
        if tag not in _metadata.get_tag_index_map():
            print('tag: %s has index: %d' % (tag, _metadata.get_tag_index()))
            _metadata.update_tag_index_map({tag: _metadata.get_tag_index()})
            _metadata.update_reverse_tag_index_map({_metadata.get_tag_index(): tag})
            tag_index = _metadata.get_tag_index()
            _metadata.set_tag_index(tag_index + 1)
        else:
            tag_index = _metadata.get_tag_index_map()[tag]
            # print('tag: %s has index: %d, debug:%s, %d' % (tag, tag_index, str(_metadata.get_tag_index_map()), _metadata.get_tag_index()))

        token_buffer = [self.tokenizer.bos_token_id]
        loss_mask_buffer = [0]

        conversations = example[self.config.conversation_key]

        for conv in conversations:
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            human_tokens = self.tokenizer.encode(human, add_special_tokens=False) if human else []
            assistant_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

            input_tokens = self.inst_begin_tokens + human_tokens + self.inst_end_tokens if human_tokens else []
            output_tokens = assistant_tokens + [self.tokenizer.eos_token_id]

            token_buffer += input_tokens + output_tokens
            loss_mask_buffer += [0] * len(input_tokens) + [1] * len(output_tokens)

        return token_buffer, loss_mask_buffer, [tag_index] * len(token_buffer), *aux


class HuggingfaceDataset(object):
    """ Huggingface dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = 'c4'
        config.name = 'en'
        config.split = 'train'
        config.streaming = False
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.batch_token_dtype = 'i4'
        config.throughput_average_window_size = 200
        config.dataset_sample_prob = ''

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        if ',' not in self.config.path:
            self._dataset = load_dataset(
                self.config.path, name, split=split, streaming=self.config.streaming
            )
        else:
            paths = self.config.path.split(',')
            if name:
                name = name.split(',')
            else:
                name = [''] * len(paths)
            if split:
                split = split.split(',')
            else:
                split = [''] * len(paths)
            datasets = []
            for i in range(len(paths)):
                datasets.append(load_dataset(
                    paths[i], name[i] if i < len(name) else None, split=split[i] if i < len(split) else None, streaming=self.config.streaming
                ))
            prob = list(map(float, self.config.dataset_sample_prob.split(','))) if self.config.dataset_sample_prob else None
            self._dataset = interleave_datasets(datasets, probabilities=prob)
            

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        total_tokens = 0
        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = total_tokens
        while True:
            token_buffer = []
            loss_mask_buffer = []
            for index, example in enumerate(self._dataset):
                tokens, loss_masks = self.text_processor(example)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_masks)
                while len(token_buffer) > chunk_size + 1:
                    total_tokens += chunk_size
                    step_times.append(time.time() - last_time)
                    last_time = time.time()
                    if len(step_times) > self.config.throughput_average_window_size:
                        step_times = step_times[-self.config.throughput_average_window_size:]
                    average_throughput = chunk_size / np.mean(step_times)
                    accumulated_throughput = (
                        (total_tokens - start_tokens) / (time.time() - start_time)
                    )
                    metrics = {
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                        'dataset_accumulated_tps': accumulated_throughput,
                        'dataset_average_tps': average_throughput,
                    }
                    batch = {
                        'input_tokens': np.array(token_buffer[:chunk_size], dtype=self.config.batch_token_dtype).reshape(
                            self.config.batch_size, -1
                        ),
                        'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=self.config.batch_token_dtype).reshape(
                            self.config.batch_size, -1
                        ),
                        'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        ),
                    }
                    if self.config.always_start_with_bos:
                        batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                    yield batch, metrics
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(config=self.config)

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class JsonDataset(object):
    """ JSON dataset, where each line of the data file contains a JSON
        dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200
        config.enable_padding = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self._total_tokens = self.config.tokens_count_at_start
        self.total_tag_tokens = dict()

    def parse_json(self, line):
        if not line or line == '\n':
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            return None
        return data

    def json_iterator(self):
        with mlxu.open_file(self.config.path, 'r') as fin:
            fin.seek(self._file_loc)
            while True:
                line = fin.readline()
                self._file_loc = fin.tell()
                if not line:   # Reached EOF
                    self._index = 0
                    fin.seek(0)
                    continue

                data = self.parse_json(line)
                if data is not None:
                    # JSON parsing succeeded
                    yield data, self._file_loc, self._index
                self._index += 1

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index in self.json_iterator():
                yield self.text_processor((example, loc, index), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(
                self.json_iterator(), self.config.tokenizer_parallel_batch_size
            )
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn, next(batched_iterator),
                    chunksize=self.config.tokenizer_parallel_chunk_size
                )
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn, next(batched_iterator),
                        chunksize=self.config.tokenizer_parallel_chunk_size
                    )
                    for example in current_batch.get():
                        yield example

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        token_buffer = []
        loss_mask_buffer = []
        tag_buffer = []
        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        row_token_buffer = []
        for tokens, loss_masks, langs, loc, index in self.parallel_example_iterator():
            if self.config.enable_padding and len(tokens) + len(row_token_buffer) > self.config.seq_length:
                n_remain = self.config.seq_length - len(row_token_buffer)
                if n_remain > 0:
                    token_buffer.extend([self.tokenizer.eos_token_id] * n_remain)
                    loss_mask_buffer.extend([0] * n_remain)
                    # TODO: Deal with this tag problem for padding.
                    tag_buffer.extend([0] * n_remain)
                row_token_buffer = []

            token_buffer.extend(tokens)
            if self.config.enable_padding:
                row_token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            tag_buffer.extend(langs)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += chunk_size

                target_langs_buffer = tag_buffer[1:chunk_size + 1]
                for i in range(_metadata.tag_index.value):
                    if ('dataset_%s_tokens' % _metadata.get_reverse_tag_index_map()[i]) not in self.total_tag_tokens:
                        self.total_tag_tokens['dataset_%s_tokens' % _metadata.get_reverse_tag_index_map()[i]] = 0
                    cnt = 0
                    for j in target_langs_buffer:
                        if j == i:
                            cnt += 1
                    self.total_tag_tokens['dataset_%s_tokens' % _metadata.get_reverse_tag_index_map()[i]] += cnt

                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size:]
                average_throughput = chunk_size / np.mean(step_times)
                accumulated_throughput = (
                    (self._total_tokens - start_tokens) / (time.time() - start_time)
                )
                metrics = {
                    'dataset_file_loc': loc,
                    'dataset_example_index': index,
                    'dataset_total_tokens': self._total_tokens,
                    'dataset_accumulated_tps': accumulated_throughput,
                    'dataset_average_tps': average_throughput,
                }
                metrics.update(self.total_tag_tokens)
                batch = {
                    'input_tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'target_tags': np.array(tag_buffer[1:chunk_size + 1], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                        self.config.batch_size, -1
                    ),
                }
                if self.config.always_start_with_bos:
                    batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                yield batch, metrics
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]
                tag_buffer = tag_buffer[chunk_size:]
                if self.config.enable_padding:
                    row_token_buffer = token_buffer.copy()

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
            total_tokens=self._total_tokens,
            total_tag_tokens=self.total_tag_tokens
        )

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))
        self._index = state_dict.get('index', self.config.example_index_at_start)
        self._file_loc = state_dict.get('file_loc', self.config.start_seek_loc)
        self._total_tokens = state_dict.get('total_tokens', self.config.tokens_count_at_start)
        self.total_tag_tokens = state_dict.get('total_tag_tokens', dict())

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)
