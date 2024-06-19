from datasets import load_dataset
from nltk.probability import FreqDist
from transformers import AutoTokenizer
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

    ### CONSTANTS ###
    __USC_STR = "United States Code"

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
    # MORE EXAMPLES: section 54AA, section 1400Z-1
    __section_ref = r'\b[Ss]ections?(?>\s?(?>(?>\d{1,4}(?>\w{1,2}-?\d?)(?>\(\w{1,3}\))*)|(?>(?>(?>and)|(?>or)))),?)+'

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

    __ref_patterns = {
        # External reference patterns
        # "title": __title_ref, 
        # "subtitle": __subtitle_ref,
        # "chapter:": __chapter_ref,
        # "subchapter": __subchapter_ref,
        # "part": __part_ref,
        # "subpart": __subpart_ref,
        "section": __section_ref,

        #Internal reference patterns
        # "subsection": __subsection_ref,
        # "paragraph": __paragraph_ref,
        # "subparagraph": __subparagraph_ref,
        # "clause": __clause_ref,
        # "subclause": __subclause_ref,
        # "item": __item_ref,
        # "subitem": __subitem_ref,
        # "subsubitem": __subsubitem_ref
    }

    def __init__(self, filepath) -> None:
        """Constructor for UscDatasetBuilder class. It loads a huggingface
        Dataset

        Args:
            filepath (str): Path to huggingface Dataset file.
        """
        self._ds = load_dataset("csv", data_files=filepath, split='train')
        # self._ds = self._ds.select(range(4))

        # create a dict to map to index of the matrix where I'll count references
        # to other sections
        self._section_index_dict = dict(
            list(zip(self._ds["level_name"], range(1, len(self._ds["level_name"]))))
        )

        self.__logfile = open("output/logfile.txt", "w")

            # this matrix will contain counts of references to other sections
        self._ext_ref_matrix = None

    def __del__(self):
        self.__logfile.close()

# this function extracts extra text around the section reference to help me
# further analyze what the reference refers to.
## NOTE: consider adding a label argument so I can log how I'm categorizing it
## (i.e., USC or non-USC section)
    def __log_reference_context(self, x, pretext, start, end):
        s = pretext + " --> " + x['level'] + " " + x['level_name'] + " (" + str(start) + "," + \
            str(end) + "): " + x['text'][start-20:end+50] + '\n'
        self.__logfile.write(s)
        
    # protected members

    # public members
    def __get_section_index(self, section):
        if(section not in self._section_index_dict.keys()):
            return None
        return self._section_index_dict[section]

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

        assert(y)

        if cased:
            self._tokenizer = AutoTokenizer.from_pretrained(
                "google-bert/bert-base-cased",
                use_fast = True
            )
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                "google-bert/bert-base-uncased",
                use_fast = True
            )

        self._ds = self._ds.map(
            lambda x: {'tokens': self._tokenizer.tokenize(x['text'])},
            num_proc=num_proc
        )

    def _calc_shannon_entropy(self, tokens):

        ## count occurrences of tokens
        fd = FreqDist(tokens) 

        ## convert counts to probabilities
        values = [v for v in fd.values()]
        probs = values/np.sum(values)

        return entropy(probs)

    ## TODO: implement a version of "add_shannon_entropy" based on unique words
    ## so use str.split()
    def add_shannon_entropy(self) -> None:
        """Calculates the Shannon Entropy from the subword tokens created from
        UscDatasetBuilder.add_tokens and inserts the results in the 'entropy'
        column of the Dataset.

        NOTE: The entropy of the tokens will be different from the entropy
        of the actual words, because--depending on the tokenizer--the tokens
        will include stemmed words like "sect#". 
        """
        assert("tokens" in self._ds.column_names)

        self._ds = self._ds.map(
            lambda x: {'entropy': self._calc_shannon_entropy(x['tokens'])}
            )
        # print(self._ds['entropy'])

    # Function purpose - extract and categorize section references
    # for each match found in the seciton text, test to see if it is a reference that points
    # to an external USC title. Possible categorizations include:
    # 1) External USC title
    # 2) External non-USC reference
    # 3) Internal reference to current USC section? 
    def _extract_references(self, x, ref_pattern):
        search_window_size = 20

        # find matches
        # matches = re.findall(ref_pattern, x['text'])
        matches = []
        p = re.compile(ref_pattern)

        for m in p.finditer(x['text']):
            ref = x['text'][m.start():m.end()]

            refs = self._split_reference([ref])
            if(refs == []):
                continue

            for ref in refs:
                
                # var to label reference type
                ref_type = ""

                # get target section number
                section_num = re.search("\d{1,4}", ref ).group()

                # looking for "United States Code" to begin within search_window_size characters 
                # of the end of the section reference
                end_of_context = m.end() + search_window_size + len(self.__USC_STR)

                # extract extra text after "section" to analyze what the section
                # reference points to (i.e., another USC section or another non-USC document)
                context_string = x['text'][m.start():end_of_context]
                
                # If "title" is found, then it's an external reference (just need to categorize it as
                # a specific type of external reference)
                # NOTE: assuming "title" appears after section reference; there may be instances where
                # the info identifying the target of the reference appears before "section"

                # check if "title" is found in context_string, if so 
                # EXTERNAL REFERENCE
                if(re.search("[Tt]itle \d{1,4}",context_string)):
                    # EXTERNAL REFERENCE
                    if(context_string.find(self.__USC_STR) >= 0):
                        # EXTERNAL REFERENCE to another USC Title
                        # >~95% of the section references I analyzed that pointed to another title in the USC
                        # had something like "title 49, United States Code[,.]".
                        ref_type = "EXT_USC"

                    else:
                        # if the wording "such Code" follows the reference, then assume it's referring to
                        # the same document (and therefore is of the same reference type) as the 
                        # prior reference.  NOTE: this is not perfect, but hopefully better than leaving this
                        # code out
                        if(context_string.find("such Code") >= 0):
                            # SAME AS PRIOR REFERENCE
                            ref_type = matches[-1][1]
                        else:
                            # EXTERNAL REFERENCE to a non-USC document
                            ref_type = "EXT_NON_USC"
                else:
                    # INTERNAL REFERENCE
                    # Check if it exists in __section_index_dict
                    ref_type = "INT"

                    if(not self.__get_section_index(section_num)):
                        ref_type = "UNKOWN"

                # only log reference if it does not point to USC
                self.__log_reference_context(x, ref_type, m.start(), m.end())
            matches.append((ref, ref_type))

        if(not matches):
            return []

        return flatten_list([matches])

    def add_references(self):

        for level, pattern in self.__ref_patterns.items():

            self._ds = self._ds.map(
                lambda x: {level: self._extract_references(x, pattern)}
            )

    def _split_reference(self, ref_list):
        refs = []

        if (len(ref_list) == 0):
            return refs

        for ref in ref_list:
            ## the ref_list is not completely pure "section [number]" strings
            ## we only want "section [number]" strings so when we count the
            ## number of section references, we can just take find len(x['section'])
            exclude = [
                ref == "section and",
                ref == "section and,",
                ref == "section or",
                ref == "section or,",
                ref == 'section'
            ]
            if any(exclude):
                continue
            refs += [s.strip() for s in re.split("\s+or|,|and\s+", ref)]

        # remove empty items and "and" and "or" items
        l = list(filter(lambda x: (x != '' and x!="and" and x!="or"), refs))
        return l

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

    ## DON'T NEED THIS METHOD ANYMORE -- 'add tokens just splits into
    ## word tokens now.
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

    def _count_external_references(self, x):

        current_section_index = self.__get_section_index(x['level_name'])

        ## process the list of section references extracted (which is contained in x['section'])
        for sectionref in x['section']:
            ## 1) match section number in the reference, there should only be
            ## one match per list item
            
            # pattern to match section number with possible subsection
            m = re.search("\d{1,4}(?>\w{1,2}-?\d?)", sectionref)

            ## every item in the section column should contain one and only
            ## one section reference
            assert(len(m) == 1) 

            start, end = m.span()
            sectionnum = sectionref[start:end]


            ## 2) get referenced section index
            ref_section_index = self.__get_section_index(sectionnum)

            if (not ref_section_index):
                # report that the section reference was not found in the dictionary
                print(
                    "Does not exist: " + sectionref + " in " + x["level"] + " " + x["level_name"]
                )
                # move onto next reference
                continue

            ## 3) use current section index and referenced section index to
            ## increment correct position in matrix
            self._ext_ref_matrix[current_section_index, ref_section_index] += 1

    def create_ext_ref_matrix(self):
        ## add_references() needs to have been called first
        if ("section" not in self._ds.column_names):
            self.add_references()

        ## initialize the matrix
        n = len(self._section_index_dict)
        self._ext_ref_matrix = np.zeros((n,n), dtype=int)

        self._ds.map(self._count_external_references)

    def get_external_ref_matrix(self):
        return self._ext_ref_matrix

    def _get_word_lengths(self, text):
        return [len(word) for word in text.split()]

    def add_avg_word_length(self) -> None:
        self._ds = self._ds.map(
            lambda x: {"avg_word_length": np.mean(self._get_word_lengths(x['text']))}
        )

    def add_avg_token_length(self) -> None:
        """Adds a column ("avg_token_length") that contains the average word 
        token length). Word length is a simple measure of complexity."""

        assert("tokens" in self._ds.column_names)
        self._ds = self._ds.map(
            lambda x: {"avg_token_length": np.mean([len(word) for word in x['tokens']])}
        )

    def get_num_rows(self):
        """Returns the number of rows in the Dataset

        Returns:
            int: Number of rows.
        """
        return self._ds.num_rows

    def add_size(self) -> None:

        self._ds = self._ds.map(
            lambda x: {"size": len(x['text'])}
        )

    def add_num_words(self) -> None:

        self._ds = self._ds.map(
            lambda x: {"num_words": len([word for word in x['text'].split()])}
        )

    def save(self, json_filepath):
        self._ds.to_json(json_filepath)

ds = UscDatasetBuilder("output/usc26_sections.csv")
# ds.add_tokens()
# ds.add_avg_word_length()
# ds.add_size()
# ds.add_num_words()
# ds.add_shannon_entropy()
# ds.add_word_tokens()
# ds.add_avg_token_length()
ds.add_references()
# ds.create_ext_ref_matrix()
ds.save("output/usc26_sections_modified.json")
# print(ds._section_index_dict['1400Z–1'])
