/*
 * granite_tokenizer.c - Decode-only BPE tokenizer for Granite Speech 5.0
 */

#include "granite_tokenizer.h"
#include "granite.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct granite_tokenizer {
    char **pieces;      /* pieces[id], NUL-terminated; NULL if id unused */
    int *lens;          /* byte length of each piece */
    int vocab_size;
};

/* ------------------------------------------------------------------ JSON --- */

/* Decode a JSON string literal starting at *p (which points just past the
 * opening quote). Writes into `out` (caller-sized >= the literal length) and
 * advances *p past the closing quote. Returns the byte length written. */
static int json_unescape(const char **p, char *out) {
    const char *s = *p;
    int n = 0;
    while (*s && *s != '"') {
        if (*s != '\\') { out[n++] = *s++; continue; }
        s++;
        switch (*s) {
        case 'n': out[n++] = '\n'; s++; break;
        case 't': out[n++] = '\t'; s++; break;
        case 'r': out[n++] = '\r'; s++; break;
        case 'b': out[n++] = '\b'; s++; break;
        case 'f': out[n++] = '\f'; s++; break;
        case '/': out[n++] = '/';  s++; break;
        case '"': out[n++] = '"';  s++; break;
        case '\\': out[n++] = '\\'; s++; break;
        case 'u': {
            unsigned cp = 0;
            for (int i = 1; i <= 4; i++) {
                char c = s[i];
                cp <<= 4;
                if (c >= '0' && c <= '9') cp |= (unsigned)(c - '0');
                else if (c >= 'a' && c <= 'f') cp |= (unsigned)(c - 'a' + 10);
                else if (c >= 'A' && c <= 'F') cp |= (unsigned)(c - 'A' + 10);
            }
            s += 5;
            /* Surrogate pair. */
            if (cp >= 0xD800 && cp <= 0xDBFF && s[0] == '\\' && s[1] == 'u') {
                unsigned lo = 0;
                for (int i = 2; i <= 5; i++) {
                    char c = s[i];
                    lo <<= 4;
                    if (c >= '0' && c <= '9') lo |= (unsigned)(c - '0');
                    else if (c >= 'a' && c <= 'f') lo |= (unsigned)(c - 'a' + 10);
                    else if (c >= 'A' && c <= 'F') lo |= (unsigned)(c - 'A' + 10);
                }
                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    s += 6;
                }
            }
            if (cp < 0x80) {
                out[n++] = (char)cp;
            } else if (cp < 0x800) {
                out[n++] = (char)(0xC0 | (cp >> 6));
                out[n++] = (char)(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                out[n++] = (char)(0xE0 | (cp >> 12));
                out[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                out[n++] = (char)(0x80 | (cp & 0x3F));
            } else {
                out[n++] = (char)(0xF0 | (cp >> 18));
                out[n++] = (char)(0x80 | ((cp >> 12) & 0x3F));
                out[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                out[n++] = (char)(0x80 | (cp & 0x3F));
            }
            break;
        }
        default: out[n++] = *s++; break;
        }
    }
    *p = *s == '"' ? s + 1 : s;
    return n;
}

/* Find the `"vocab":` object that lives inside `"model":`. The file also has a
 * top-level "added_tokens" array but no other "vocab" key, so the last match is
 * unambiguous; we search from the "model" key to be safe. */
static const char *find_vocab(const char *json) {
    const char *model = strstr(json, "\"model\"");
    const char *from = model ? model : json;
    const char *v = strstr(from, "\"vocab\"");
    if (!v) return NULL;
    v = strchr(v, '{');
    return v ? v + 1 : NULL;
}

granite_tokenizer_t *granite_tokenizer_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "granite: cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) { fclose(f); return NULL; }

    char *json = malloc((size_t)size + 1);
    if (!json) { fclose(f); return NULL; }
    if (fread(json, 1, (size_t)size, f) != (size_t)size) {
        free(json); fclose(f); return NULL;
    }
    json[size] = '\0';
    fclose(f);

    const char *p = find_vocab(json);
    if (!p) {
        fprintf(stderr, "granite: no model.vocab in %s\n", path);
        free(json);
        return NULL;
    }

    granite_tokenizer_t *t = calloc(1, sizeof(*t));
    if (!t) { free(json); return NULL; }
    t->vocab_size = GRANITE_VOCAB;
    t->pieces = calloc((size_t)t->vocab_size, sizeof(char *));
    t->lens = calloc((size_t)t->vocab_size, sizeof(int));
    if (!t->pieces || !t->lens) {
        granite_tokenizer_free(t); free(json); return NULL;
    }

    /* Entries are `"token": id` pairs; a token is never longer than the raw
     * literal, so the literal length is a safe scratch bound. */
    char *scratch = malloc((size_t)size);
    if (!scratch) { granite_tokenizer_free(t); free(json); return NULL; }

    int loaded = 0;
    while (*p && *p != '}') {
        while (*p && *p != '"' && *p != '}') p++;
        if (*p != '"') break;
        p++;
        int len = json_unescape(&p, scratch);

        while (*p && *p != ':') p++;
        if (*p != ':') break;
        p++;
        while (*p == ' ') p++;
        long id = strtol(p, (char **)&p, 10);

        if (id >= 0 && id < t->vocab_size && !t->pieces[id]) {
            char *piece = malloc((size_t)len + 1);
            if (piece) {
                memcpy(piece, scratch, (size_t)len);
                piece[len] = '\0';
                t->pieces[id] = piece;
                t->lens[id] = len;
                loaded++;
            }
        }
        while (*p == ' ' || *p == ',') p++;
    }

    free(scratch);
    free(json);

    if (loaded == 0) {
        fprintf(stderr, "granite: empty vocab in %s\n", path);
        granite_tokenizer_free(t);
        return NULL;
    }
    return t;
}

void granite_tokenizer_free(granite_tokenizer_t *t) {
    if (!t) return;
    if (t->pieces) {
        for (int i = 0; i < t->vocab_size; i++) free(t->pieces[i]);
        free(t->pieces);
    }
    free(t->lens);
    free(t);
}

int granite_tokenizer_vocab_size(const granite_tokenizer_t *t) {
    return t ? t->vocab_size : 0;
}

/* `<0xHH>` byte-fallback pieces decode to the raw byte HH. */
static int byte_fallback(const char *piece, int len, unsigned char *out) {
    if (len != 6 || piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' ||
        piece[5] != '>')
        return 0;
    unsigned v = 0;
    for (int i = 3; i <= 4; i++) {
        char c = piece[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return 0;
    }
    *out = (unsigned char)v;
    return 1;
}

char *granite_tokenizer_decode(const granite_tokenizer_t *t,
                               const int *ids, int n_ids) {
    if (!t) return NULL;

    size_t cap = 256, len = 0;
    char *out = malloc(cap);
    if (!out) return NULL;

    for (int i = 0; i < n_ids; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->vocab_size || !t->pieces[id]) continue;
        const char *piece = t->pieces[id];
        int plen = t->lens[id];

        if (len + (size_t)plen + 1 > cap) {
            while (len + (size_t)plen + 1 > cap) cap *= 2;
            char *nb = realloc(out, cap);
            if (!nb) { free(out); return NULL; }
            out = nb;
        }

        unsigned char b;
        if (byte_fallback(piece, plen, &b)) {
            out[len++] = (char)b;
            continue;
        }
        /* Replace U+2581 (E2 96 81) with a space. */
        for (int j = 0; j < plen; j++) {
            if (j + 2 < plen && (unsigned char)piece[j] == 0xE2 &&
                (unsigned char)piece[j + 1] == 0x96 &&
                (unsigned char)piece[j + 2] == 0x81) {
                out[len++] = ' ';
                j += 2;
            } else {
                out[len++] = piece[j];
            }
        }
    }

    /* Strip(content=" ", start=1): drop a single leading space. */
    size_t start = (len > 0 && out[0] == ' ') ? 1 : 0;
    memmove(out, out + start, len - start);
    out[len - start] = '\0';
    return out;
}
