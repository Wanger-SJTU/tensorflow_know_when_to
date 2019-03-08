import os
import numpy as np
import heapq


class CaptionData(object):
    def __init__(self, sentence, memory, output, alpha, score):
       self.sentence = sentence
       self.memory = memory
       self.output = output
       self.alpha = alpha
       self.score = score

    def __cmp__(self, other):
        assert isinstance(other, CaptionData)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    def __lt__(self, other):
        assert isinstance(other, CaptionData)
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, CaptionData)
        return self.score == other.score

class TopN(object):
    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        self._data = []


class CaptionGenerator(object):
    def __init__(self,
                model,
                vocabulary,
                beam_size = 3,
                max_caption_length=20,
                batch_size = 32):
        self.model = model
        self.vocabulary = vocabulary
        self.beam_size = beam_size
        self.max_caption_length = max_caption_length
        self.batch_size = batch_size

    def beam_search(self, sess, images,vocabulary):
        """Use beam search to generate the captions for a batch of images."""
        # Feed in the images to get the contexts and the initial LSTM states

        contexts, initial_memory, initial_output = sess.run(
            [self.model.reshaped_conv5_3_feats, self.model.initial_memory, self.model.initial_output],
            feed_dict = {self.model.images: images})

        partial_caption_data = []
        complete_caption_data = []
        for k in range(self.batch_size):
            initial_alpha_k = [np.zeros([self.model.num_ctx],dtype=np.float32)]
            initial_beam = CaptionData(sentence = [],
                                       memory = initial_memory[k],
                                       output = initial_output[k],
                                       alpha = initial_alpha_k,
                                       score = 1.0)
            partial_caption_data.append(TopN(self.beam_size))
            partial_caption_data[-1].push(initial_beam)
            complete_caption_data.append(TopN(self.beam_size))

        # Run beam search
        for idx in range(self.max_caption_length):
            partial_caption_data_lists = []
            for k in range(self.batch_size):
                data = partial_caption_data[k].extract()
                partial_caption_data_lists.append(data)
                partial_caption_data[k].reset()

            num_steps = 1 if idx == 0 else self.beam_size
            for b in range(num_steps):
                if idx == 0:
                    last_word = np.zeros((self.batch_size), np.int32)
                else:
                    last_word = np.array([pcl[b].sentence[-1]
                                        for pcl in partial_caption_data_lists],
                                        np.int32)

                last_memory = np.array([pcl[b].memory
                                        for pcl in partial_caption_data_lists],
                                        np.float32)
                last_output = np.array([pcl[b].output
                                        for pcl in partial_caption_data_lists],
                                        np.float32)

                memory, output, scores, alpha = sess.run(
                    [self.model.memory, self.model.output, self.model.probs, self.model.alpha],
                    feed_dict = {self.model.contexts: contexts,
                                 self.model.last_word: last_word,
                                 self.model.last_memory: last_memory,
                                 self.model.last_output: last_output})
                ''' 
                # For Debug
                aaaa = sess.run(
                    fetches='attend/Softmax:0',
                    feed_dict= {'Placeholder_1:0': contexts,
                                'Placeholder_3:0': last_output}
                    )

                haha = 100
                '''

                # Find the beam_size most probable next words
                for k in range(self.batch_size):
                    caption_data = partial_caption_data_lists[k][b]
                    words_and_scores = list(enumerate(scores[k]))
                    words_and_scores.sort(key=lambda x: -x[1])
                    words_and_scores = words_and_scores[0:self.beam_size+1]

                    # Append each of these words to the current partial caption
                    for w, s in words_and_scores:
                        sentence = caption_data.sentence + [w]
                        score = caption_data.score * s
                        alpha_sent = caption_data.alpha +[alpha[k]]
                        beam = CaptionData(sentence,
                                           memory[k],
                                           output[k],
                                           alpha_sent,
                                           score)
                        if vocabulary.words[w] == '.':
                            complete_caption_data[k].push(beam)
                        else:
                            partial_caption_data[k].push(beam)

        results = []
        for k in range(self.batch_size):
            if complete_caption_data[k].size() == 0:
                complete_caption_data[k] = partial_caption_data[k]
            results.append(complete_caption_data[k].extract(sort=True))

        return results


