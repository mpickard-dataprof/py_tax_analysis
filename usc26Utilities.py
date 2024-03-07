import pandas as pd
import numpy as np

df = pd.read_csv("./output/usc26.csv")

def countLevels(df):
    levels = df['level'].unique()
    counts = [(level, len(df[df['level']==level])) for level in levels]
    df = pd.DataFrame(counts)
    df.columns = ["level", "counts"]
    df = df[~df['level'].isin(['continuation', 'content', 'chapeau'])]
    return df.set_index('level')

def removeDuplicateText(df):
    """A utility function that removes text that duplicate the text of 
    parent levels. The USC Title 26 XML file has <content>, <continuation>,
    and <chapeau> tags that are not logically children of the tags they
    reside in -- by that, I mean they are not true "levels" of the USC
    document. This function removes those duplicate levels and returns
    the deduplicated dataframe.

    Args:
        df (pandas.DataFrame): The Title 26 dataframe generated from parsing
        the XML file. It needs to have 'id', 'parent', 'level' and 'text'
        columns.

    Returns:
        pandas.DataFrame: A dataframe with the duplicated levels dropped.
    """
    # parent-to-child is 1-to-many. So 'id' acts as the parent identifier
    # and 'parent' (which is the parent id of the child) acts as the child
    # identifier.  This allows a parent to be matched with many children.
    # (Subtitle A has 7 Chapter children)
    merged_df = df.merge(df, how='right', left_on='id', right_on='parent')

    # create a df containing rows that have the same parent and child text, 
    # where the child is a 'continuation', 'chapeau', or 'content'.
    # The logic here is if a continuation, chapeau, or content node (which
    # are "helper" nodes, not actually logical children) AND it duplicates
    # it's parent's text; then delete it.

    # IMPORTANT: there are instances in real level parent-child relationships
    # where there is the same text. One example include '/us/usc/t26/s101/f/2/A'.
    # It has the same text as it's parent '/us/usc/t26/s101/f/2'.  We want to
    # keep these...that's why we ensure the child is a continuation, chapeau, 
    # or a content.
    sametext_df = merged_df[
        (merged_df['text_x'] == merged_df['text_y']) &
        (merged_df['level_y'].isin(['continuation', 'chapeau', 'content']))
        ]

    # extract the children index ids that are duplicates
    drop_index = sametext_df['row_y'].values

    # drop them from the original df
    return df.drop(drop_index)

def get_num_words(df):
    # need a "text" column
    assert(df.text)
    # TODO: find a way to assert that it's a string column

    df['numWords'] = [len(str(text).split()) for text in df.text]
    return df


# NOTE: It was orders of magnitude faster to generate the section
# dataframe from the XML (see usc_xml_to_df.py) rather than reassemble 
# the parts of the section like I attempt here with combine_sections()
    
# def combine_sections(df):
#     """Combines text of subparts of a section into one string that is 
#     equivalent to the complete text of the section. Returns a dataframe
#     with the section id, id of the section parent, and the full section
#     text.

#     Args:
#         df (pandas.DataFrame): expecting the dataframe returned by
#         UscReader.XmltoDataframe(), which contains a level id, parent id,
#         and text.
#     """
#     # assert that the df contains the needed columns
#     assert('text' in df.columns)
#     assert('id' in df.columns)
#     assert('parent' in df.columns)

#     # get unique lists of sections and their parents
#     section_list = df[df['level'] == 'section'].id.unique()
#     parent_list = df[df['level'] == 'section'].parent

#     # iterate through the section list
#     # for each section, find all rows that belong in that section

#     section_text_list = []    
#     for section in section_list:
#         # pattern to match the exact section.
#         # specifically, had to handle cases such as not matching
#         # '/us/usc/t26/s1563/' or '/us/usc/t26/s1563/f/2/B' 
#         # when searching for '/us/usc/t26/s1' (i.e., section 1)
#         section_regex = r'^' + section + r'(?:(?:\/)?|(?:\/.*))$'

#         # some ids (such as "content" and "continuation" levels) are "NaN", 
#         # so search for regex in "parent" field as well. A match in the
#         # "parent" field for NaN levels means their text belongs in that section
#         filtered_df = df[
#             df['id'].str.contains(section_regex, na=False) | 
#             df['parent'].str.contains(section_regex)
#             ]
        
#         # join the parts of the section together and add it
#         # to the list of section texts
#         section_text_list.append(
#             # a ". " separate would break up "real" sentences --
#             # for instance, chapeaus which end in "-" (so would 
#             # "-."), so I opted for a " " separator.
#             # no good solution when trying to look forward to the 
#             # effects on sentence parsing.
#             filtered_df['text'].str.cat(sep=' ')
#         )

#     new_df = pd.DataFrame.from_dict(
#         {
#             "section": section_list,
#             "parent": parent_list,
#             "text": section_text_list
#         }
#     )

#     new_df.to_csv("output/section_df.csv")

# print(countLevels(df))
# print(countLevels(removeDuplicateText(df)))
# combine_sections(df)
