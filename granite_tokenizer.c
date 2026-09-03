/* Decode-only tokenizer for Granite Speech 5.0 CTC output. */

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


/* Unescape a JSON string and advance p past its closing quote. */
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

/* Find the vocab object inside the model object. */
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

    /* The unescaped token fits within its JSON literal. */
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

/*
 * GPT-2 byte-level BPE maps every raw byte to a unicode codepoint (see
 * `bytes_to_unicode` in HF's tokenizers): printable ASCII/Latin-1 bytes map
 * to themselves, and the rest (control chars, space, etc.) map to codepoints
 * starting at U+0100. Notably 0x20 (space) maps to U+0120 ('Ġ'). Vocab
 * pieces are therefore UTF-8 encodings of those codepoints, not raw text,
 * and must be mapped back through the inverse table to recover the bytes.
 */
static short g_byte_of_cp[512];
static int g_table_built = 0;

static void build_byte_unicode_table(void) {
    unsigned short cp_of_byte[256];
    int assigned[256] = {0};
    for (int b = 33; b <= 126; b++) { cp_of_byte[b] = (unsigned short)b; assigned[b] = 1; }
    for (int b = 161; b <= 172; b++) { cp_of_byte[b] = (unsigned short)b; assigned[b] = 1; }
    for (int b = 174; b <= 255; b++) { cp_of_byte[b] = (unsigned short)b; assigned[b] = 1; }
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!assigned[b]) {
            cp_of_byte[b] = (unsigned short)(256 + n);
            n++;
        }
    }
    for (int i = 0; i < 512; i++) g_byte_of_cp[i] = -1;
    for (int b = 0; b < 256; b++) g_byte_of_cp[cp_of_byte[b]] = (short)b;
    g_table_built = 1;
}

/* Decode one UTF-8 codepoint from `s`, writing its length to *adv. */
static unsigned decode_utf8_cp(const char *s, int remaining, int *adv) {
    unsigned char c0 = (unsigned char)s[0];
    if (c0 < 0x80) { *adv = 1; return c0; }
    if ((c0 & 0xE0) == 0xC0 && remaining >= 2) {
        *adv = 2;
        return ((unsigned)(c0 & 0x1F) << 6) | ((unsigned)(unsigned char)s[1] & 0x3F);
    }
    if ((c0 & 0xF0) == 0xE0 && remaining >= 3) {
        *adv = 3;
        return ((unsigned)(c0 & 0x0F) << 12) |
               (((unsigned)(unsigned char)s[1] & 0x3F) << 6) |
               ((unsigned)(unsigned char)s[2] & 0x3F);
    }
    *adv = 1;
    return c0;
}

char *granite_tokenizer_decode(const granite_tokenizer_t *t,
                               const int *ids, int n_ids) {
    if (!t) return NULL;
    if (!g_table_built) build_byte_unicode_table();

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

        for (int j = 0; j < plen; ) {
            int adv;
            unsigned cp = decode_utf8_cp(piece + j, plen - j, &adv);
            short b = cp < 512 ? g_byte_of_cp[cp] : -1;
            out[len++] = (char)(b >= 0 ? (unsigned char)b : (unsigned char)cp);
            j += adv;
        }
    }

    /* Strip(content=" ", start=1): drop a single leading space. */
    size_t start = (len > 0 && out[0] == ' ') ? 1 : 0;
    memmove(out, out + start, len - start);
    out[len - start] = '\0';
    return out;
}
