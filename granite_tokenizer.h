/*
 * granite_tokenizer.h - Decode-only BPE tokenizer for Granite Speech 5.0
 *
 * CTC never encodes text, so only the id -> string direction is implemented:
 * no merges, no normalizer, no pre-tokenizer.
 */

#ifndef GRANITE_TOKENIZER_H
#define GRANITE_TOKENIZER_H

typedef struct granite_tokenizer granite_tokenizer_t;

/* Load the `model.vocab` table out of a tokenizers-format tokenizer.json. */
granite_tokenizer_t *granite_tokenizer_load(const char *path);
void granite_tokenizer_free(granite_tokenizer_t *t);

/* Decode ids to UTF-8, applying the checkpoint's decoder sequence:
 * Replace("▁" -> " "), ByteFallback, Fuse, Strip(one leading space).
 * Returns a malloc'd string the caller frees. */
char *granite_tokenizer_decode(const granite_tokenizer_t *t,
                               const int *ids, int n_ids);

int granite_tokenizer_vocab_size(const granite_tokenizer_t *t);

#endif /* GRANITE_TOKENIZER_H */
