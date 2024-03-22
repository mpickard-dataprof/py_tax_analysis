from datasets import load_dataset
from transformers import BertTokenizer
from scipy.stats import entropy
import numpy as np
import regex as re

## UTILITY FUNCTIONS
def flatten_list(nested_list):
    flat_list = []
    for row in nested_list:
        flat_list += row
    return flat_list

class UscDatasetBuilder:

    # NOTE: These regex patterns use atomic groups which are not natively
    # supported in Python until 3.11. <3.11 need to import `regex` (not `re`).

    ### EXTERNAL REFERENCE regex definitions ###

    # EXAMPLES: Title 9 * Title 18a * titles 9 and 10 * Titles 9, 12a, 13, or 15
    __title_ref = r'\b[Tt]itles?(?>\s?(?>(?>\d{1,4}\w{0,2})|(?>(?>(?>and)|(?>or)))),?)+'

    # EXAMPLES: Subtitle A * subtitles A or B * Subtitles A, B, C * subtitles A, C, D, E, or F
    __subtitle_ref = r'\b[Ss]ubtitles?(?>\s(?>(?>\b[A-K]\b)|(?>(?>(?>and)|(?>or)))),?)+'

    # EXAMPLES: Chapter 2A * chapter 100 * Chapters 6, 7, 8 * chapters 6A, 7, 72, or 10
    __chapter_ref = r'\b[Cc]hapters?(?>\s?(?>(?>\d{1,4}\w?)|(?>(?>(?>and)|(?>or)))),?)+'

    # EXAMPLES: subchapter A * subchapters A, B, C * Subchapters A, B, C, or F * Subchapter A and B
    __subchapter_ref = r'\b[Ss]ubchapters?(?>\s?(?>(?>\b\w\b)|(?>(?>(?>and)|(?>or)))),?)+'

    # EXAMPLES: part II * parts I, II, or IV * parts II and I * part II and III
    __part_ref = r'\b[Pp]arts?(?>\s?(?>(?>\b[A-Z]{1,4}\b)|(?>(?>(?>and)|(?>or)))),?)+'

    # EXAMPLES: Subpart A * subparts A, D, E * Subparts A, D, and Z * subpart A or B
    __subpart_ref = r'\b[Ss]ubparts?(?>\s?(?>(?>\b[A-Z]\b)|(?>(?>(?>and)|(?>or)))),?)+'

    # EXAMPLES: section 2503 * section 2012(d) * section 2010(c)(3) * Sections 2432(d)(3), 2351(c), or 1123
    __section_ref = r'\b[Ss]ections?(?>\s?(?>(?>\d{1,4}?(?>\(\w{1,3}\))*)|(?>(?>(?>and)|(?>or)))),?)+'

    __external_refs = [__title_ref, __subtitle_ref, __chapter_ref, __subchapter_ref,
                       __part_ref, __subpart_ref, __section_ref]

    ### INTERNAL REFERENCES regex definitions ###

    # EXAMPLES: subsection (d) * Subsection (e)(1)(C) * subsections (b)(1), (c), (e)(1)(C) * subsection (b)(1) or (b)(2)
    __subsection_ref = r'\b[Ss]ubsections?(?>\s?(?>(?>\(\w{1,3}\),?)|(?>(?>(?>and)|(?>or)))))+'

    # EXAMPLES: paragraph (2) * paragraphs (1)(B) and (2)(C) * paragraphs (1)(C), (2)(G)(iii), and (5) * paragraph (1)(A) and (3)(F)
    __paragraph_ref = r'\b[Pp]aragraphs?(?>\s?(?>(?>\(\w{1,3}\))|(?>(?>(?>and)|(?>or)))),?)+'

    # EXAMPLES: subparagraph (A) * Subparagraph (A)(i)(I) * Subparagraph (B)(iii) or (D)(i)(I) * subparagraph (A) or subparagraph (B)
    __subparagraph_ref = r'\b[Ss]ubparagraphs?(?>\s?(?>(?>\(\w{1,4}\))|(?>(?>(?>and)|(?>or)))),?)+'

    # EXAMPLES: Clause (i) * clause (iii)(III)(aa) * clauses (ii)(III)(ab), (i)(IV), (i) * clauses (i), (ii)(ab)(III), (iv)(I) and (iii)
    __clause_ref = r'\b[Cc]lauses?(?>\s?(?>(?>\(\w{1,4}\))|(?>(?>(?>and)|(?>or)))),?)+'

    # EXAMPLES: Subclause (III)(ab)(AB) * subclause (I) or (II) * Subclauses (I), (V), (IX), and (X) * subclauses (I) and (IV)
    __subclause_ref = 'r\b[Ss]ubclauses?(?>\s?(?>(?>\(\w{1,4}\))|(?>(?>(?>and)|(?>or)))),?)+'

    # EXAMPLES: item (bb)(AB)(aaa) * item (ac)(AB)(aaf) and (df)(BB) * Items (ai)(AC), (ak)(BE)(ffe), (ak), or (am) * items (aa) and (ab)
    __item_ref = 'r\b[Ii]tems?(?>\s?(?>(?>\(\w{1,4}\))|(?>(?>(?>and)|(?>or)))),?)+'

    # EXAMPLES: subitem (AB)(aaa) * subitem (AC) and (AD)(aac) * Subitems (DE), (AL)(aae), and (AB) * subitems (AA) and (AB)
    __subitem_ref = r'\b[Ss]ubitems?(?>\s?(?>(?>\(\w{1,4}\))|(?>(?>(?>and)|(?>or)))),?)+'

    # EXAMPLES: Subsubitem (aaa) * Subsubitems (aac), (aaa), (aac) * Subsubitems (eei), (ffe), (bff), or (aae) * Subsubitems (aad) or (aal)
    __subsubitem_ref = r'\b[Ss]ubsubitems?(?>\s?(?>(?>\(\w{1,4}\))|(?>(?>(?>and)|(?>or)))),?)+'

    __internal_refs = [__subsection_ref, __paragraph_ref, __subparagraph_ref, __clause_ref,
                       __subclause_ref, __item_ref, __subitem_ref, __subsubitem_ref]

    def __init__(self, filepath) -> None:
        self._ds = load_dataset("csv", data_files=filepath, split='train')
        # self._ds = self._ds.select(range(4))
        self._cased = True

    # protected members

    # public members
    def add_tokens(self, cased=True) -> None:
        if self._cased:
            self._tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-base-cased"
            )
        else:
            self._tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-base-uncased"
            )

        self._ds = self._ds.map(
            lambda row: self._tokenizer(row["text"], return_tensors="np"), 
            batched=True,
            num_proc=12
        )

    def add_shannon_entropy(self):
        self._ds = self._ds.map(lambda x: {'entropy': entropy(x['input_ids'])})
        # print(self._ds['entropy'])

    def _extract_references(self, text, type):
        assert(type == "internal" or type == "external")
        if (type == "internal"):
            ref_patterns = self.__internal_refs
        else: 
            ref_patterns = self.__external_refs

        matches = [re.findall(regex, text) for regex in ref_patterns]
        matches = flatten_list(matches)

        ## second pass to clean up "or", "and" or commas at the end of a reference
        matches = [
            match.removesuffix("or").removesuffix("and").removesuffix(",").strip()
            for match in matches
        ]
        return matches

    def add_internal_references(self):
        self._ds = self._ds.map(
            lambda x: {"internal_refs": self._extract_references(x['text'], "internal")}
        )

        # print(self._ds['internal_refs'])

    def add_external_references(self):
        self._ds = self._ds.map(
            lambda x: {"external_refs": self._extract_references(x['text'], "external")}
        )

        # print(self._ds['external_refs'])

    def add_word_tokens(self):
        self._ds = self._ds.map(
            lambda x: {"tokens": self._tokenizer.convert_ids_to_tokens(x["input_ids"])}
        )
        # print(self._ds['tokens'][1])

    ## NOTE: tokens are abbreviated and stemmed...so, may need to split words to get
    ## true average word length.
    def add_avg_token_length(self):
        self._ds = self._ds.map(
            lambda x: {"avg_word_length": np.mean([len(word) for word in x['tokens']])}
        )
        # print(self._ds['avg_word_length'])

    def get_num_rows(self):
        return self._ds.num_rows

ds = UscDatasetBuilder("output/usc26_sections.csv")
# ds.add_tokens()
# ds.add_shannon_entropy()
# ds.add_word_tokens()
# ds.add_avg_token_length()
ds.add_internal_references()
ds.add_external_references()
