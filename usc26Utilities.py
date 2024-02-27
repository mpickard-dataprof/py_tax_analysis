import pandas as pd

# df = pd.read_csv("./output/usc26.csv")

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

# print(countLevels(df))
# print(countLevels(removeDuplicateText(df)))
