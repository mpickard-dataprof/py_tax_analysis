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

    ### --- NOTES ---- ###
    # Made deliberate decision to focus on internal and external with respect
    # to sections, so:
    #    - External references are section level and above
    #    - Internal references are subsection level and below
    # Matching single (ex. Title 18) and compound multiple references (Subtitles A, B, and C)
    #   - thus compound multiple references count as "many" references
    # The first word does not always identify the type of reference. For example,
    # in Section 1:
    #    - "section 2(a)" is a reference to a subsection, not a section
    #    - "subparagraph (A)(ii)" refers to a clause
    #    - "sections 63(c)(4) and 151(d)(4)(A))" refers to a paragraph and
    #      a subparagraph
    # so...will need to take this into account if we ever want a count 
    # references to specific internal levels (e.g., subsection, paragraph, etc).
    # The regex patterns were complicated and weren't completely sufficient--for
    # example, the matches would have trailing commas, "or", and "and"--so I 
    # took "after-match passes" to clean them up.

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
        """Constructor for UscDatasetBuilder class. It loads a huggingface
        Dataset

        Args:
            filepath (str): Path to huggingface Dataset file.
        """
        self._ds = load_dataset("csv", data_files=filepath, split='train')
        # self._ds = self._ds.select(range(4))

    # protected members

    # public members
    def add_tokens(self, cased=False, num_proc=6) -> None:
        """Creates a huggingface Tokenizer and adds a column to the Dataset
        with a list of the tokens from the Tokenizer. The tokens are extract
        from the text column of the Dataset.

        Args:
            cased (bool, optional): Determines whether the Tokenizer should
            be case-sensitive or not. Defaults to False.
            num_proc (integer, optional): number of processes to use. Defaults 
            to 6.
        """

        assert("text" in self._ds.column_names)

        if cased:
            self._tokenizer = BertTokenizer.from_pretrained(
                "spacy/en_core_web_md"
            )
        else:
            self._tokenizer = BertTokenizer.from_pretrained(
                "spacy/en_core_web_md"
            )

        self._ds = self._ds.map(
            lambda row: self._tokenizer(row["text"], return_tensors="np"), 
            batched=True,
            num_proc=num_proc
        )

    ## TODO: implement a version of "add_shannon_entropy" based on unique words
    ## so use str.split()
    def add_shannon_entropy(self) -> None:
        """Calculates the Shannon Entropy from the numeric tokens created from
        UscDatasetBuilder.add_tokens and inserts the results in the 'entropy'
        column of the Dataset.

        NOTE: The entropy of the tokens will be different from the entropy
        of the actual words, because--depending on the tokenizer--the tokens
        will include stemmed words like "sect#". 
        """
        assert("input_ids" in self._ds.column_names)

        self._ds = self._ds.map(lambda x: {'entropy': entropy(x['input_ids'])})
        # print(self._ds['entropy'])

    def _extract_references(self, text, type) -> list[str]:
        """A utility function to find and return all the USC references--
        either "external" or "internal" references--in the text provided.

        Args:
            text (str): A string of text to match USC references in.
            type (str): Specifics whether to match "internal" or "external" 
            references. "internal" uses the list of regexes in 
            UscDatasetBuilder.__internal_refs. "external" uses the list of
            regexes in UscDatasetBuilder.__external_refs.

        Returns:
            list[str]: A list of strings that are the matches references.
        """
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

    def add_internal_references(self) -> None:
        """Adds a column to the Dataset with a list of the matched internal
        references found in the section text.
        """

        assert("text" in self._ds.column_names)
        self._ds = self._ds.map(
            lambda x: {"internal_refs": self._extract_references(x['text'], "internal")}
        )

        # print(self._ds['internal_refs'])

    def add_external_references(self) -> None:
        """Adds a column to the Dataset with a list of the matched external
        references found in the section text.
        """

        assert("text" in self._ds.column_names)
        self._ds = self._ds.map(
            lambda x: {"external_refs": self._extract_references(x['text'], "external")}
        )

        # print(self._ds['external_refs'])

    def add_num_external_refs(self):
        """Adds a column with the number of external references found in the section. 
        These are references external to the section (i.e., section and above).
        """
        if("external_refs" not in self._ds.column_names):
            self.add_external_references()
        
        self._ds = self._ds.map(
            lambda x: {"num_ext_refs": len(x["external_refs"])}
        )
            
    def add_num_internal_refs(self):
        """Adds a colum with the number of internal references found in the section. 
        These are references internal to the section (i.e., subsection and below).
        """
        if("internal_refs" not in self._ds.column_names):
            self.add_internal_references()
        
        self._ds = self._ds.map(
            lambda x: {"num_int_refs": len(x["internal_refs"])}
        )
            

    def add_word_tokens(self) -> None:
        """Adds a column with the word tokens. The tokenizer used in add_tokens()
        just adds a column with numeric tokens. Functions like add_avg_token_length()
        need the actual words.

        NOTE: the word tokens are stemmed (depending on the tokenizer used), so
        "average word length" may not be accurate.
        """
        assert("input_ids" in self._ds.column_names)
        self._ds = self._ds.map(
            lambda x: {"tokens": self._tokenizer.convert_ids_to_tokens(x["input_ids"])}
        )
        # print(self._ds['tokens'][1])

    ## NOTE: tokens are abbreviated and stemmed...so, may need to split words to get
    ## true average word length.
    def add_avg_token_length(self) -> None:
        """Adds a column ("avg_word_length") that contains the average word 
        token length). Word length is a simple measure of complexity."""

        assert("tokens" in self._ds.column_names)
        self._ds = self._ds.map(
            lambda x: {"avg_word_length": np.mean([len(word) for word in x['tokens']])}
        )
        # print(self._ds['avg_word_length'])

    def get_num_rows(self):
        """Returns the number of rows in the Dataset

        Returns:
            int: Number of rows.
        """
        return self._ds.num_rows

    def save(self, path):
        self._ds.to_csv(path)

ds = UscDatasetBuilder("output/usc26_sections.csv")
ds.add_tokens()
ds.add_shannon_entropy()
ds.add_word_tokens()
ds.add_avg_token_length()
ds.add_internal_references()
ds.add_external_references()
ds.add_num_internal_refs()
ds.add_num_external_refs()
ds.save("output/usc26_sections_modified.csv")
