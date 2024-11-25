from benchmark.runs.run import Run


class Warmup(Run):
    def start (self):
        for i in range(self.passes):
            input_ids = self.tokenizer.encode(self.dataset.get_item(i), return_tensors='pt')
            output = self.model.generate(input_ids, max_length=60, num_return_sequences=5, no_repeat_ngram_size=2)

            for sequence in output:
                generated_text = self.tokenizer.decode(sequence, skip_special_tokens=True)
                print(generated_text)

