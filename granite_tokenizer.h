/* Decode-only tokenizer for Granite Speech 5.0 CTC output. */

#ifndef GRANITE_TOKENIZER_H
#define GRANITE_TOKENIZER_H

typedef struct granite_tokenizer granite_tokenizer_t;

/* Load the `model.vocab` table out of a tokenizers-format tokenizer.json. */
granite_tokenizer_t *granite_tokenizer_load(const char *path);
void granite_tokenizer_free(granite_tokenizer_t *t);

/* Decode token IDs to a malloc'd UTF-8 string; caller frees it. */
char *granite_tokenizer_decode(const granite_tokenizer_t *t,
                               const int *ids, int n_ids);

int granite_tokenizer_vocab_size(const granite_tokenizer_t *t);

#endif
